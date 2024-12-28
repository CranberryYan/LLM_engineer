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

#include <iostream>
#include "src/kernels/fused_transpose_and_remv_pad.h"

// [bs, head_num, seq_len, head_size] -> [bs, seq_len, head_num, head_size] 
//  -> [num_tokens, head_num, head_size]
// 说明在bs, seq_len中有一些是padding出来的, remove后, 重新回到输入的num_tokens
template<typename T>
__global__ void fused_transpose_reshape_remv_pad(
    T *src, T *dst, const int num_tokens, const int batch_size,
    const int seq_len, const int head_num, const int head_size,
    const int *padding_offset)
{
    int token_id = blockIdx.x;
    int batch_id = (token_id + padding_offset[token_id]) / seq_len;
    int seq_id = (token_id + padding_offset[token_id]) % seq_len;

    // transpose前后的offset
    int src_offset = batch_id * head_num * seq_len * head_size + seq_id * head_size;
    int dst_offset = token_id * head_num * head_size;
    
    for (int i = threadIdx.x; i < head_num *head_size; i += blockDim.x) {
        int head_id = i / head_size;
        int head_siez_id = i % head_size;
        dst[dst_offset + i] = src[src_offset + head_id * seq_len * head_size + head_siez_id];
    }
}

template <typename T>
void launchTransposeOutRemovePadding(
    TensorWrapper<T> *qkv_buf_w_pad,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<T> *qkv_buf_wo_pad_1) {
    int batch_size = qkv_buf_w_pad->shape[0];
    int head_num = qkv_buf_w_pad->shape[1];
    int seq_len = qkv_buf_w_pad->shape[2];
    int head_size = qkv_buf_w_pad->shape[3];
    int num_tokens = qkv_buf_wo_pad_1->shape[0];

    dim3 grid(num_tokens);
    dim3 block(std::min(head_num * head_size, 1024));
    fused_transpose_reshape_remv_pad<T><<<grid, block>>>(
        qkv_buf_w_pad->data, qkv_buf_wo_pad_1->data,
        num_tokens, batch_size, seq_len, head_num, head_size,
        padding_offset->data);
}

template void launchTransposeOutRemovePadding(
    TensorWrapper<float> *qkv_buf_w_pad,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<float> *qkv_buf_wo_pad_1);
template void launchTransposeOutRemovePadding(
    TensorWrapper<half> *qkv_buf_w_pad,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<half> *qkv_buf_wo_pad_1);