// input: [num_tokens] -> input_embedding: [num_tokens, hidden_size]
//                              |
//                              -> cal_paddingoffset: [bs, max_num_tokens, hidden_size]
//                              |
//                              -> build_casual_mask: mask: [bs, max_num_tokens, max_num_tokens]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_size] -> fusedQkvGemm: * [hidden_size, hidden_size] -> [num_tokens, hidden_size]
//                              -> AddbiasAndPaddingAndRope: [max_num_tokens, hidden_size] -> [bs, q_head_num, max_q_len, head_size]  ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            -> ConcatPastKVcache: [num_layers, bs, kv_head_num, max_seq_len(8192), head_size]
//                                            |     cache的内容: [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
//                                            |
//                                            -> Broadcast: kv: [bs, q_head_num, max_k_len, head_size]
//								-> Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
//                              -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len, hidden_size](bs*seq_len = num_tokens)
//                                  -> [num_tokens, hidden_size]
//                              -> FusedAddbiasResidual: [num_tokens, hidden_size]
#include <stdio.h>
#include "src/kernels/fused_addresidual_norm.h"


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

// 1.this kernel is used after self attention in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
// residual.shape = [num tokens, hidden_units]
template<typename T>
__global__ void FusedAddBiasResidualRMSNorm(
    T* residual,   // int&out: [num tokens, q_hidden_units]
    T* decoder_in, // int&out: [num tokens, q_hidden_units]
    /*optional*/T* bias,  // [hidden_units]
    T* scale,             // [hidden_units], RMSNorm weights
    float eps, int num_tokens, int hidden_units) {
    int batch_id = blockIdx.x;
    int vec_size = Vec<T>::size; // size: static, 无需将类实例化, 可以调用, 2 or 4
    using Vec_t = typename Vec<T>::Type;

    // 每个block负责一行, 每一行有hidden_units个元素
    // dout: 每一行的第一个元素
    Vec_t *de_out = reinterpret_cast<Vec_t*>(decoder_in + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);
    Vec_t *bia;
    if (bias != nullptr) {
        bia = reinterpret_cast<Vec_t*>(bias);
    }

    Vec_t tmp;
    T thread_sum = static_cast<T>(0.0f);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        // 向量化, idx每次加1, 但是因为de_out是float4 or half2, 因此地址会默认每次加vec_size
        // idx: [0, hidden_units / vec_size] -> [0, hidden_units]
        if (residual != nullptr) {
            de_out[idx].x += rsd[idx].x;
            de_out[idx].y += rsd[idx].y;
            de_out[idx].z += rsd[idx].z;
            de_out[idx].w += rsd[idx].w;

            rsd[idx].x = de_out[idx].x;
            rsd[idx].y = de_out[idx].y;
            rsd[idx].z = de_out[idx].z;
            rsd[idx].w = de_out[idx].w;
        }
        if (bias != nullptr) {
            de_out[idx].x += bia[idx].x;
            de_out[idx].y += bia[idx].y;
            de_out[idx].z += bia[idx].z;
            de_out[idx].w += bia[idx].w;
        }
        thread_sum += de_out[idx].x * de_out[idx].x;
        thread_sum += de_out[idx].y * de_out[idx].y; 
        thread_sum += de_out[idx].z * de_out[idx].z; 
        thread_sum += de_out[idx].w * de_out[idx].w; 
    }

    // mean(x^2)
    T block_sum = blockReduceSum<float>(thread_sum);
    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf(block_sum / hidden_units + eps);
    }
    __syncthreads();

    // rmsnorm
    if (scale != nullptr) {
        Vec_t* s = reinterpret_cast<Vec_t*>(scale);
        for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
            de_out[idx].x = de_out[idx].x * inv_mean * s[idx].x;
            de_out[idx].y = de_out[idx].y * inv_mean * s[idx].y;
            de_out[idx].z = de_out[idx].z * inv_mean * s[idx].z;
            de_out[idx].w = de_out[idx].w * inv_mean * s[idx].w;
        }
    }
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm(
    TensorWrapper<T> *residual, 
    TensorWrapper<T> *decoder_in,
    BaseWeight<T>& norm, T* scale, float eps) {
    T *bias = norm.bias;
    int num_tokens = decoder_in->shape[0];
    int hidden_units = decoder_in->shape[1];
    int num_thread = std::min<int>(hidden_units / 4, 1024); // assume head size can be divided by 4 and 2
    
    dim3 grid(num_tokens);
    dim3 block(num_thread);
    printf("calling FusedAddResidualAndRMSNorm\n");
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>
        (residual->data, decoder_in->data, bias,
        scale, eps, num_tokens, hidden_units);
    printf("called FusedAddResidualAndRMSNorm\n");
}

// residual.shape = [num tokens, hidden_units]
template void launchFusedAddBiasResidualRMSNorm(
    TensorWrapper<float> *residual, 
    TensorWrapper<float> *decoder_in, // [num tokens, hidden_units]
    BaseWeight<float> &norm,
    float* scale, //RMSNorm weights
    float eps);
