# include "src/kernels/build_casual_mask.h"


/*
仅考虑当前轮次的上下文
100
110
111
下三角: 0: 不应该被看的"未来"

考虑所有轮次的上下文
111  100
111  110
111  111
上文全部为1: 全应该被考虑, 当前轮次同上
q: 3(当前序列)   k: 6(全部上下文)
*/

// 主要就是判断那些地方是 1 or 0
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(
    T* mask,            // mask: [bs, max_q_len, max_k_len]
    const int* q_lens,  // input lens: [bs]
    const int* k_lens,  // context lens: [bs]
    int max_q_len, int max_k_len // 当前轮次最大的 q or k
    )
{
    int tid = threadIdx.x;

    // blockIdx.x: 当前batch(句子)
    int qlen = q_lens[blockIdx.x]; // 当前batch(句子)的行数
    int klen = k_lens[blockIdx.x]; // 当前batch(句子)的列数

    mask += blockIdx.x * max_q_len * max_k_len; // 分别处理每个batch(句子)
    while (tid < max_q_len * max_k_len) {
        int q = tid / max_k_len; // 行
        int k = tid % max_k_len; // 列
        bool if_one = q < qlen && k < klen    // 边界判断
                            &&
                      k <= q + (klen - qlen); // 1的位置(考虑上下文) 
        mask[tid] = static_cast<T>(if_one);

        tid += blockDim.x;
    }
}


template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask,     
        TensorWrapper<int>* q_lens,TensorWrapper<int>* k_lens)
{
    int batch_size  = mask->shape[0];
    int max_q_len   = mask->shape[1];
    int max_k_len   = mask->shape[2];
    
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>
        (mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}

// 模板函数实例化
// FP32
template void launchBuildCausalMasks(TensorWrapper<float>* mask,
                TensorWrapper<int>* q_lens, TensorWrapper<int>* k_lens);

// FP16
template void launchBuildCausalMasks(TensorWrapper<half>* mask,
                TensorWrapper<int>* q_lens, TensorWrapper<int>* k_lens);