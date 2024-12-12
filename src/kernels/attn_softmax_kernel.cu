// Attention mask + scale + softmax:
//  Attention mask: causal mask(下三角矩阵)
//  Scale: sqrt(head size)
//  Softmax: softmax
// input: Qk gemm
// Qk gemm -> Attention mask -> Scale -> Softmax
// 三个访存密集算子的融合

// input: [M, N]
// 每个block处理一行(N个元素)
// 
#include <math.h>
#include <float.h>
#include <iostream>
#include <assert.h>
# include "attn_softmax_kernel.h"


template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpReduce(T val) {
    for (int mask = 32 / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T blockReduce(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ T warp[64];
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0) {
        warp[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warp[tid] : 0;
    
    return warpReduce<ReductionOp, T>(warp_val);
}

template <typename T_half, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax_half(T_half *attn_score, T_half *qk, T_half *mask,
    int batch_size, int head_nums,
    int q_len, int k_len, float scale) {

}

template <typename T, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax_float(T *attn_score, T *qk, T *mask,
	int batch_size, int head_nums,
	int q_len, int k_len, float scale) {
	
	// dim3 grid(batch_size, head_nums, q_length);
	// dim3 block((k_length + 32 - 1) / 32 * 32);
	int batch_id = blockIdx.x;
	int head_id = blockIdx.y;
	int q_id = blockIdx.z;
	if (threadIdx.x >= k_len) {
		return;
	}
	// qk: [batch_size, head_num, q_length, k_legnth]
	// attention_mask:  [batch_size, q_length, k_length]
	__shared__ float s_max;
	__shared__ float inv_sum;
	for (int i = q_id; i < q_len; i += gridDim.z) { // 行
		int qk_offset = 0;
		int mask_offset = 0;
		T thread_max = FLT_MIN;
		T thread_sum = 0.0f;
		T max_val	= FLT_MIN;
		T sum_val = 0.0f;
		T qk_data = static_cast<T>(0);
		T mask_data = static_cast<T>(0);
		T data[NUMS_PER_THREAD_PER_ROW];
		for (int j = 0; j < NUMS_PER_THREAD_PER_ROW; ++j) { // 列
			qk_offset = batch_id * head_nums * q_len * k_len +
									head_id * q_len * k_len +
									i * k_len + 
									j * blockDim.x + threadIdx.x;
			qk_data = qk[qk_offset];
			mask_offset = batch_id * q_len * k_len +
										i * k_len + 
										j * blockDim.x + threadIdx.x;
			mask_data = mask[mask_offset];

			data[j] = scale * qk_data + (1 - mask_data) * (-10000.0f); // 1: 可以看见   0: 不可见
			thread_max = fmax(data[j], thread_max);
		}
		max_val = blockReduce<MaxOp, T>(thread_max);
		if (threadIdx.x == 0) {
			s_max = max_val; // 行最大值
		}
		__syncthreads();
		for (int j = 0; j < NUMS_PER_THREAD_PER_ROW; ++j) {
			data[j] = expf(data[j] - s_max);
			thread_sum += data[j];
		}
		sum_val = blockReduce<SumOp, T>(thread_sum);
		if (threadIdx.x == 0) {
			inv_sum = 1 / (sum_val + 1e-6f);
		}
		__syncthreads();

		for (int j = 0; j < NUMS_PER_THREAD_PER_ROW; ++j) {
			qk_offset = batch_id * head_nums * q_len * k_len +
									head_id * q_len * k_len +
									i * k_len + 
									j * blockDim.x + threadIdx.x;
			attn_score[qk_offset] = (data[j] * inv_sum);
		}
	}
}


template<typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T>* qk, TensorWrapper<T>* mask, 
	TensorWrapper<T>* attn_score, float scale) {

	// attention_score: [batch_size, head_num, q_length, k_legnth]
	// qk:              [batch_size, head_num, q_length, k_legnth]
	// attention_mask:  [batch_size, q_length, k_length]
	int batch_size = qk->shape[0];
	int head_nums = qk->shape[1];
	int q_length = qk->shape[2];
	int k_length = qk->shape[3];
	bool is_half = sizeof(T) == 2;
	LLM_CHECK_WITH_INFO(k_length % 2 == 0, "K_len, should be divided by 4!\n");

	dim3 grid(batch_size, head_nums, q_length);
	dim3 block((k_length + 32 - 1) / 32 * 32);

	if (is_half) {
		int vec_size = 2;
		if (block.x > 2048 && block.x <= 4096) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 4;  // 每行每个线程处理4个数据
			block.x /= 4 * vec_size;                    // 没必要再分配这么多线程了
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_half<half, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(half *)attn_score->data, (half *)qk->data, (half *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		} else if (block.x > 1024) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 2;  // 每行每个线程处理2个数据
			block.x /= 2 * vec_size;                    // 没必要再分配这么多线程了
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_half<half, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(half *)attn_score->data, (half *)qk->data, (half *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		} else if (block.x <= 1024) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 1;  // 每行每个线程处理1个数据
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_half<half, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(half *)attn_score->data, (half *)qk->data, (half *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		}
	} else {
		if (block.x > 2048 && block.x <= 4096) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 4;  // 每行每个线程处理4个数据
			block.x /= 4;                   						// 没必要再分配这么多线程了
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_float<float, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(float *)attn_score->data, (float *)qk->data, (float *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		} else if (block.x > 1024) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 2;  // 每行每个线程处理2个数据
			block.x /= 2;                 							// 没必要再分配这么多线程了
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_float<float, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(float *)attn_score->data, (float *)qk->data, (float *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		} else if (block.x <= 1024) {
			constexpr int NUMS_PER_THREAD_PER_ROW = 1;  // 每行每个线程处理1个数据
			block.x = (block.x + 32 - 1) / 32 * 32;
			assert(block.x < 1024);
			ScaleMaskAndSoftmax_float<float, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>(
					(float *)attn_score->data, (float *)qk->data, (float *)mask->data,
					batch_size, head_nums, q_length, k_length, scale);
		}
	}
}

template void launchScaleMaskAndSoftmax(TensorWrapper<float> *qk, TensorWrapper<float> *mask,
    TensorWrapper<float> *attn_score, float scale);

template void launchScaleMaskAndSoftmax(TensorWrapper<half> *qk, TensorWrapper<half> *mask,
    TensorWrapper<half> *attn_score, float scale);