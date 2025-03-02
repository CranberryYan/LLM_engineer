// 归一化
// input: [num_tokens] -> input_embedding: [num_tokens, hidden_units](num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)
//                              |
//                              -> cal_paddingoffset: [bs, max_q_len, hidden_units]
//                              |
//                              -> build_casual_mask: mask: [bs, max_q_len, max_k_len]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_units]
#include <iostream>
#include "rmsnorm_kernel.h"

/*
RMSNorm公式:
    x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    1. x的平方并求均值
    2. 开平方的倒数
*/

/*
val: 当前线程的值(比如 tid 对应的值）
i: 要与哪个线程的值进行交换或相加的偏移量
传进去tid=1的值，然后i=16，将tid=1与tid=17的值相加，
然后i=8，将tid=9的值累加，i=4，i=2, i=1直到i=0退出循环,
tid=1 与 tid=17 9 5 3 2的和全部保存在tid=1

(x): 无用
0 = 0 + 16  0 = 0 + 8   0 = 0 + 4       0 = 0 + 2       0 = 0 + 1
1 = 1 + 17  1 = 1 + 9   1 = 1 + 5       1 = 1 + 3       1 = 1 + 2(x)
2 = 2 + 18  2 = 2 + 10  2 = 2 + 6       2 = 2 + 4(x)    2 = 2 + 3(x)
3 = 3 + 19  3 = 3 + 11  3 = 3 + 7       3 = 3 + 5(x)    3 = 3 + 4(x)
4 = 4 + 20  4 = 4 + 12  4 = 4 + 8(x)    4 = 4 + 6(x)    4 = 4 + 5(x)
5 = 5 + 21  5 = 5 + 13  5 = 5 + 9(x)    5 = 5 + 7(x)    5 = 5 + 6(x)
最后一次的 0 = 0 + 1, 并不是要将 1 = 1 + 2的 1 与 0 相加
如果这样做: 0 = 16 + 8 + 4 + 2 + 1 + 17 + 9 + 5 + 3 + 2, 多加了一次2
*/
template<typename T>
__device__ T warpReduceSum(T val) {
	for (int i = 32 / 2; i > 0; i >>= 1) {
		val += __shfl_xor_sync(0xffffffff, val, i);
	}

	return val;
}

// note: 防止blockDim.x < 32 -> 向上进一 
template<typename T>
__device__ T blockReduceSum(T val) {
	int tid = threadIdx.x;
	int wid = tid / 32;
	int laneid = tid % 32; // 当前tid在warp中的id
	int warpnum = (blockDim.x + 32 - 1) / 32;

	// 在laneid=0的val中, 保存每个warp的sum
	val = warpReduceSum<T>(val);

	// 报错: smem中的参数要在编译期间确定大小, 因此warpnum不行
	// static __shared__ T warpsum[warpnum];
	// shared_memory: 48KB
	// 使用64, 防止32不够
	static __shared__ T warpsum[64];
	if (laneid == 0) {
		warpsum[wid] = val;
	}
	__syncthreads();

	T sum = tid < warpnum ? warpsum[tid] : (T)0;
	sum = warpReduceSum<T>(sum);

	return sum;
}

template <typename T>
__global__ void RMSNorm(T* decoder_out, // [num tokens, q_hidden_units]
                        T* decoder_residual,
                        T* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units) {
  int vec_size = Vec<T>::size;
  using Vec_t = typename Vec<T>::Type;
  float thread_sum = 0.0f;
  Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
  Vec_t* rsd;
  rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);
  for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
    Vec_t vec = dout[idx];
    rsd[idx] = vec;
    thread_sum += vec.x * vec.x;
    thread_sum += vec.y * vec.y;
    thread_sum += vec.z * vec.z;
    thread_sum += vec.w * vec.w;
  }
  thread_sum = blockReduceSum<float>(thread_sum);
  __shared__ float inv_mean;
  if (threadIdx.x == 0) {
    inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
  }
  __syncthreads();
  Vec_t* s = reinterpret_cast<Vec_t*>(scale);
  for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
    Vec_t out = dout[idx];// note the offset should divide vec size

    dout[idx].x = out.x * inv_mean * s[idx].x;
    dout[idx].y = out.y * inv_mean * s[idx].y;
    dout[idx].z = out.z * inv_mean * s[idx].z;
    dout[idx].w = out.w * inv_mean * s[idx].w;
  }
}

template <>
__global__ void RMSNorm(half* decoder_out, // [num tokens, q_hidden_units]
                        half* decoder_residual,
                        half* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t* s; 
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);
    Vec_t* rsd;
    if (decoder_residual != nullptr) {
        rsd = reinterpret_cast<Vec_t*>(decoder_residual + batch_id * hidden_units);
    }
    float thread_accm = 0.0f;
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = dout[i];// note the offset should divide vec size
        if (decoder_residual != nullptr) {
            rsd[i] = out;
        }
        thread_accm += __half2float(out.x) * __half2float(out.x);
        thread_accm += __half2float(out.y) * __half2float(out.y);
    } //x^2
    
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrtf(float(blocksum / hidden_units) + eps);
    }
    __syncthreads();
    // rmsnorm
    s = reinterpret_cast<Vec_t*>(scale);
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t dout_h2 =dout[i];
        dout[i].x = s[i].x * __float2half(__half2float(dout_h2.x) * inv_fenmu);
        dout[i].y = s[i].y * __float2half(__half2float(dout_h2.y) * inv_fenmu);
    }    
}


template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,
                    LayerNormWeight<T>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last // for print last rmsnorm output to debug
                    ) {
	int num_tokens = decoder_out->shape[0];
	int hidden_units = decoder_out->shape[1];
	int vec_size = Vec<T>::size;
	int num_threads = hidden_units / 4; //vec size // assume head size can be divided by 4 and 2
	T* rsd = decoder_residual->data;
	dim3 grid(num_tokens);
	dim3 block(num_threads);
	RMSNorm<T><<<grid, block>>>(decoder_out->data,
													rsd,
													attn_norm_weight.gamma,
													eps,
													num_tokens,
													hidden_units);
#ifdef PRINT_DATA
	print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}

template void launchRMSNorm( TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
									TensorWrapper<float>* decoder_residual,
									LayerNormWeight<float>& attn_norm_weight, //RMSNorm weights
									float eps, //RMSNorm eps
									bool is_last);

template void launchRMSNorm( TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<half>* decoder_residual,
                    LayerNormWeight<half>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last);