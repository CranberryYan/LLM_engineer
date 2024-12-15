// 归一化
// input: [num_tokens] -> input_embedding: [num_tokens, hidden_size]
//                              |
//                              -> cal_paddingoffset: [bs, max_num_tokens, hidden_size]
//                              |
//                              -> build_casual_mask: mask: [bs, max_num_tokens, max_num_tokens]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_size]
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

template<typename T>
__global__ void RMSNorm(
    T* decoder_in,  // input & output: [num_tokens, hidden_size]
    T* scale,       // intput: [hidden_size], RMSNorm weights
    float eps,      // RMSNorm: 防止0作被除数
    int num_tokens,
    int hidden_units) {
}

template<>
__global__ void RMSNorm(float* decoder_in, float* scale,
    float eps, int num_tokens, int hidden_units) {
    int vec_size = Vec<float>::size; // size: static, 无需将类实例化, 可以调用
    using Vec_t = typename Vec<float>::Type;

    // 每个block负责一行, 每一行有hidden_units个元素
    // dout: 每一行的第一个元素
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_in + blockIdx.x * hidden_units);
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx]; // 向量化, idx每次加1, 但是因为dout是float4 or half2, 因此地址会默认每次加 4 or 2
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y; 
        thread_sum += vec.z * vec.z; 
        thread_sum += vec.w * vec.w; 
    }
    thread_sum = blockReduceSum<float>(thread_sum);

    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf(thread_sum / hidden_units + eps);
    }
    __syncthreads();

    Vec_t* s = reinterpret_cast<Vec_t*>(scale);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx];
        dout[idx].x = vec.x * inv_mean * s[idx].x;
        dout[idx].y = vec.y * inv_mean * s[idx].y;
        dout[idx].z = vec.z * inv_mean * s[idx].z;
        dout[idx].w = vec.w * inv_mean * s[idx].w;
    }
}

template<>
__global__ void RMSNorm(half* decoder_in, half* scale,
    float eps, int num_tokens, int hidden_units) 
{
    int vec_size = Vec<half>::size; // size: static, 无需将类实例化, 可以调用
    using Vec_t = typename Vec<half>::Type;

    // 每个block负责一行, 每一行有hidden_units个元素
    // dout: 每一行的第一个元素
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_in + blockIdx.x * hidden_units);
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx]; // 向量化, idx每次加1, 但是因为dout是float4 or half2, 因此地址会默认每次加 4 or 2
        thread_sum += __half2float(vec.x) * __half2float(vec.x);
        thread_sum += __half2float(vec.y) * __half2float(vec.y);
    }
    thread_sum = blockReduceSum<half>(thread_sum);

    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf(thread_sum / hidden_units + eps);
    }
    __syncthreads();

    Vec_t* s = reinterpret_cast<Vec_t*>(scale);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx];
        dout[idx].x = __float2half(__half2float(vec.x) * inv_mean * __half2float(s[idx].x));
        dout[idx].y = __float2half(__half2float(vec.y) * inv_mean * __half2float(s[idx].y));
    }
}


// input: [num_tokens, hidden_size]
template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_in,
    LayerNormWeight<T> &attn_norm_weight, float eps) {
    int num_tokens = decoder_in->shape[0];
    int hidden_units = decoder_in->shape[1];
    int num_thread = std::min<int>(hidden_units / 4, 1024); // 向量化的读取, 一个thread负责4个数据, 最大为1024
    dim3 grid(num_tokens);
    dim3 block(num_thread);
    RMSNorm<T><<<grid, block>>>(decoder_in->data, attn_norm_weight.gamma, eps, num_tokens, hidden_units);
}

template void launchRMSNorm(TensorWrapper<float>* decoder_out,
    LayerNormWeight<float> &attn_norm_weight, float eps);

template void launchRMSNorm(TensorWrapper<half>* decoder_out,
    LayerNormWeight<half> &attn_norm_weight, float eps);               
