#include <iostream>
#include "src/kernels/repeat_kv.h"


// [num_layers, bs, kv_head_num, max_seq_len, head_size] ->
//  [bs, q_head_num, max_k_len, head_size]
template <typename T>
__global__ void repeat_value_cache(T *v_dst, T *v_src,const size_t layer_offset,
    const int head_num, const int q_head_per_kv, const int head_size,
    const int *context_length, const int max_k_len, const int max_seq_len)
{
    // [bs, q_head_num, max_k_len, head_size]
    // dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num);
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto val_src = v_src + layer_offset;
    const auto val_dst = v_dst;
    const auto seq_len = context_length[batch_id];
    const int v_head_size_id = gtid % head_size;
    const int v_seq_len_id = gtid / head_size;

    if (v_seq_len_id < seq_len) {
        // head_num: q的head_num, 需要kv的head_num
        const int src_id = batch_id * (head_num / q_head_per_kv) * head_size * max_seq_len + 
            head_id / q_head_per_kv * head_size * max_seq_len +
            v_seq_len_id * head_size + 
            v_head_size_id;
        
        const int dst_id = batch_id * head_num * head_size * max_k_len + 
            head_id * head_size * max_seq_len +
            v_seq_len_id * head_size + 
            v_head_size_id;

        val_dst[dst_id] = val_src[src_id];
    }
}

template<typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src, TensorWrapper<T> *v_cache_src,
    TensorWrapper<int> *context_length, TensorWrapper<int> *layer_id,
    TensorWrapper<T> *k_cache_dst, TensorWrapper<T> *v_cache_dst)
{
    // kv cache shape: [num_layers, bs, kv_head_num, max_seq_len, head_size]
    // int batch_size = context_length->shape[0]; // 是否可以用k_cache_src->shape[1]; ???
    int batch_size = k_cache_src->shape[1];
    int kv_head_num = k_cache_src->shape[2];
    int max_seq_len = k_cache_src->shape[3];
    int head_num = k_cache_dst->shape[1];
    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];
    int layer = layer_id->getVal();

    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    int q_head_per_kv = head_num / kv_head_num;

    // kv cache shape: [num_layers, bs, kv_head_num, max_seq_len, head_size]
    int blockSize = 128;
    dim3 block(blockSize);
    dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num);

    repeat_value_cache<T><<<grid, block>>>(v_cache_dst->data, v_cache_src->data,
        layer_offset, head_num, q_head_per_kv, head_size, context_length->data,
        max_k_len, max_seq_len);

    repeat_value_cache<T><<<grid, block>>>(k_cache_dst->data, k_cache_src->data,
        layer_offset, head_num, q_head_per_kv, head_size, context_length->data,
        max_k_len, max_seq_len);
}

template void launchRepeatKVCache(TensorWrapper<float> *k_cache_src, TensorWrapper<float> *v_cache_src,
    TensorWrapper<int> *context_length, TensorWrapper<int> *layer_id,
    TensorWrapper<float> *k_cache_dst, TensorWrapper<float> *v_cache_dst);
template void launchRepeatKVCache(TensorWrapper<half> *k_cache_src, TensorWrapper<half> *v_cache_src,
    TensorWrapper<int> *context_length, TensorWrapper<int> *layer_id,
    TensorWrapper<half> *k_cache_dst, TensorWrapper<half> *v_cache_dst);