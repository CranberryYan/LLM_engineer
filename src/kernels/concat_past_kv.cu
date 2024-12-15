/*
kv_cache出现动机: 
    pagedAttention是对kv_cache所占空间的分页管理, 是典型的以内存空间换计算开销的手段
    Decoder推理中, 对于每个输入的prompt, 在计算第一个token输出的时候, 每个token的attention都要从头计算, 
但是后续的token生成中, 需要concat前面每一个token的K和V, 由于模型参数矩阵是不变的, 
此时只有刚生成的那个token的K和V需要从头计算, 所以可以把之前token的K和V缓存起来避免重复计算, 这个就叫kv_cache

显存占用大小: 
    每个decoder_layer中, 每个token的k、v矩阵都是embedding_size = num_heads * head_size, 
再乘上seqlen和batch_size, eg: bs=8, seqlen=4096, 80层layer的kv cache一共需要: 
    80 * 8192(max_seq_len) * 4096(hidden_units or num_heads * head_size) * 8 * 2Byte(FP16) = 40GB
*/

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
//                                                [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
#include <iostream>
#include "concat_past_kv.h"

// input: [bs, kv_head_num, max_q_len, head_size] 为什么这里不是max_k_len?
//  主要因为 q k v = from_tensor * qkv_weight
//  from_tensor: 输入的token经过embedding和RMSNorm后的tensor
// 造成浪费, 因为申请空间是是按照max_seq_len(8192)申请, 但是实际上的max_q_len大部分都小于max_seq_len(8192)
// kv_cache: [num_layers, bs, kv_head_num, max_seq_len(8192), head_size] ->
// [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
template <typename T>
__global__ void append_key_cache(
    T *k_dst, // [num layers, bs, kv head num, max_q_len, head size]
    const size_t layer_offset,
    const T *k_src, // [bs, kv_head num, max_q_len, head size]
    const int kv_head_num, const int head_size,
    const int *cur_query_length, const int *history_length,
    const int max_q_len, const int max_seq_len) {
    // dim3 grid(batch_size, kv_head_num, max_q_len);
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z;
    int tid = threadIdx.x;
    
    // 指针偏移到当前layer的k_cache
    // 当前layer, 写入到k_cache的那个位置
    T *k_cache_dst = k_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cum_sum_seq_len = history_length[batch_id];
    if (token_id < cur_seq_len) {
        // [bs, head_num, max_q_len, head size] -> 
        // [bs, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len + max_q_len], head size]
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + 
            head_id * max_q_len * head_size + 
            token_id * head_size + tid;

        // seqlen[history_len : history_len + max_q_len]
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size + 
            head_id * max_seq_len * head_size + 
            (cum_sum_seq_len + token_id) * head_size + tid;

        k_cache_dst[dst_offset] = k_src[src_offset];
    }
}

template <typename T>
__global__ void append_value_cache(
    T *v_dst,
    const size_t layer_offset,
    const T *v_src,
    const int kv_head_num, const int head_size,
    const int *cur_query_length, const int *history_length,
    const int max_q_len, const int max_seq_len) {
    // dim3 grid(batch_size, kv_head_num, max_q_len);
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z;
    int tid = threadIdx.x;

    // seqlen[history_len : history_len + max_q_len]
    int cum_sum_seq_len = history_length[batch_id];

    // 当前layer, 写入到k_cache的那个位置
    T *v_cache_dst = v_dst + layer_offset;

    int cur_seq_len = cur_query_length[batch_id];
    if (token_id < cur_seq_len) {
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + 
            head_id * max_q_len * head_size + 
            token_id * head_size + tid;

        // seqlen[history_len : history_len + max_q_len]
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size + 
            head_id * max_seq_len * head_size + 
            (cum_sum_seq_len + token_id) * head_size + tid;

        v_cache_dst[dst_offset] = v_src[src_offset];
    }
}

// after RoPE: k/v: [bs, kv_head_num, max_q_len, head_size]
template <typename T>
void launchConcatKVCache(
    TensorWrapper<T> *k_src,              // from qkv bias and rope
    TensorWrapper<T> *v_src,
    TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
    TensorWrapper<int> *cur_query_length, // current epoch or local input length, [batchsize]
    TensorWrapper<int> *history_length,
    TensorWrapper<T> *k_dst, TensorWrapper<T> *v_dst) {
    int layer = layer_id->getVal();
    int batch_size = k_src->shape[0];
    int kv_head_num = k_src->shape[1];
    int max_q_len = k_src->shape[2];
    int head_size = k_src->shape[3];
    int max_seq_len = k_dst->shape[2];
    int block_size = head_size; //hed_size: 目前大模型一般是 256 or 128,  
    size_t layer_offset = layer * batch_size * kv_head_num * head_size * max_seq_len;
    
    // 将前三维全部分配给gridDim, 专心考虑head_size该如何处理
    dim3 grid(batch_size, kv_head_num, max_q_len);
    append_key_cache<T><<<grid, block_size>>>(
        k_dst->data, layer_offset, k_src->data, kv_head_num, head_size,
        cur_query_length->data, history_length->data,
        max_q_len, max_seq_len);

    append_value_cache<T><<<grid, block_size>>>(
        v_dst->data, layer_offset, v_src->data, kv_head_num, head_size,
        cur_query_length->data, history_length->data,
        max_q_len, max_seq_len);

    printf("max_seq_len: %d\n", max_seq_len);
    printf("max_q_len: %d\n", max_q_len);
}

template void launchConcatKVCache(
    TensorWrapper<float> *k_src,          // from qkv bias and rope
    TensorWrapper<float> *v_src,
    TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
    TensorWrapper<int> *cur_query_length, // current epoch or local input length, [batchsize]
    TensorWrapper<int> *history_length,
    TensorWrapper<float> *k_dst,
    TensorWrapper<float> *v_dst);

template void launchConcatKVCache(
    TensorWrapper<half> *k_src,           // from qkv bias and rope
    TensorWrapper<half> *v_src,
    TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
    TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
    TensorWrapper<int> *history_length,
    TensorWrapper<half> *k_dst,
    TensorWrapper<half> *v_dst);