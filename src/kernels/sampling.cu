// 此时已经topK, 因此输入为[bs, K](降序, [bs, 0]为max)
// |-----|---|--|-|
// 最大的值, 肯定占比大, 但并不绝对, 随机值更可能落在最大的值的那一段, 但是也有可能落在最小值的那一段
// 提高多样性

#include "sampling.h"


template<typename T>
__global__ void SamplingKernel(int *topk_id, T *topk_val, 
    int *output_id, int *seqlen, bool *is_finished, 
    int K, int rand_num, int end_id, int vocab_size) {

    if (is_finished[blockIdx.x]) {
        return;
    }
    // intupt: [bs, K]
    int batch_id = blockIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = batch_id * K + tid;
    T max_val = topk_val[batch_id * K + 0];
    topk_val[offset] = expf(topk_val[offset] - max_val); // inplace, 输出保存在输入, 没有新申请buffer
    __shared__ float sum, threadhold;
    sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < K; ++i) {
            sum += (float)topk_val[i + batch_id * K]; // 当前block(bs)的sum 
        }
        curandState_t state;
        curand_init((unsigned long long)rand_num, (unsigned long long)bid, 
            (unsigned long long)0, &state);
        threadhold = curand_uniform(&state) * sum;

        output_id[batch_id] = topk_id[batch_id * K + 0] % vocab_size; // 防止某种情况不进入for循环, 初始化为最大的那一段
        for (int i = 0; i < K; ++i) {
            threadhold = threadhold - (float)topk_val[batch_id * K + i];
            if (threadhold < 0) {
                output_id[batch_id] = topk_id[batch_id * K + i] % vocab_size; // 防御性取余
                break;
            }
        }
        seqlen[bid] += is_finished[bid] ? seqlen[bid] : 1 + seqlen[bid];
        is_finished[bid] = output_id[bid] == end_id ? 1 : 0;
    }
}

// input: [bs, K]
template<typename T>
void launchSampling(TensorWrapper<int>* topk_id,    // [bs, K]
    TensorWrapper<T> *topk_val,                     // [bs, K]
    TensorWrapper<int>* seqlen,                     // [bs] bs个句子, bs个句子长度
    TensorWrapper<bool>* is_finished,               // [bs]
    TensorWrapper<int>* output_id,                  // [bs]
    IntDict& params) {

    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"];
    int step = params["step"];
    int end_id = params["end_id"];

    dim3 grid(batch_size);
    dim3 block(K);
    SamplingKernel<<<grid, block>>>(topk_id->data, topk_val->data, 
        output_id->data, seqlen->data, is_finished->data, 
        K, step, end_id, vocab_size);
}

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<float>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<half>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);