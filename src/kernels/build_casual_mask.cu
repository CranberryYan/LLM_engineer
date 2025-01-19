// 防止模型"作弊"

// input: [num_tokens] -> input_embedding: [num_tokens, hidden_units](num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)
//                              |
//                              -> cal_paddingoffset: [bs, max_q_len, hidden_units]
//                              |
//                              -> build_casual_mask: mask: [bs, max_q_len, max_k_len]
#include "src/kernels/build_casual_mask.h"

/*
context decoder: 
    mask在该阶段的CalPaddingOffset后被创建, 在AttentionMask被使用
    训练阶段: 所有句子全部输入, 并行训练(batch), 防止模型"作弊"
    推理阶段: 和训练比较像, 输入是一整个句子, 在处理前面的token, 不希望看到后面的token
self decoder: 
    没有mask, 因为此处是自回归模式, 一个接一个token输出, 输入是前面token的总和, 天然的看不到接下来的token
*/
/*
仅考虑当前轮次(当前对话)的上下文
100
110
111
下三角: 0: 不应该被看的"未来"

考虑所有轮次(所有对话历史, 不同的batch_id)的上下文
bs_0 bs_1
111  100
111  110
111  111
上文全部为1: 全应该被考虑, 当前轮次同上
q: 3(当前序列)   k: 6(全部上下文)
*/

// mask: [bs, max_q_len, max_k_len]
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(
    T *mask,            // mask: [bs, max_q_len, max_k_len]
    const int *q_lens,  // input: lens: [bs]
    const int *k_lens,  // context: lens: [bs]
    int max_q_len, int max_k_len // 当前轮次最大的 q or k
) {
    int tid = threadIdx.x;

    // 每个block处理一行 -> blockIdx.x: 当前batch(句子)
    int qlen = q_lens[blockIdx.x]; // 当前batch(句子)的行数
    int klen = k_lens[blockIdx.x]; // 当前batch(句子)的列数

    /*
    考虑所有轮次(所有对话历史, 不同的batch_id)的上下文
    bs_0 bs_1
    111  100
    111  110
    111  111
    */
    mask += blockIdx.x * max_q_len * max_k_len; // 分别处理每个batch(句子)
    for (int i = tid; i < max_q_len * max_k_len; i += blockDim.x) {
        int q = i / max_k_len; // 行
        int k = i % max_k_len; // 列
        bool if_one = q < qlen && k < klen &&
                        k <= q + (klen - qlen);
        mask[i] = static_cast<T>(if_one);
    }
}

// mask: [bs, max_q_len, max_k_len]
template<typename T>
void launchBuildCausalMasks(TensorWrapper<T> *mask,     
        TensorWrapper<int>* q_lens,TensorWrapper<int>* k_lens) {
    int batch_size  = mask->shape[0];
    int max_q_len   = mask->shape[1];
    int max_k_len   = mask->shape[2];
    
    // 每个block处理一行
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