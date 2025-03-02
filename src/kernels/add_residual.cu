#include <stdio.h>
#include "src/kernels/add_residual.h"

template<typename T>
__global__ void AddResidual(T *residual, T *decoder_out,
    int num_tokens, int hidden_units) {
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout[i].x += rsd[i].x;
        dout[i].y += rsd[i].y;
        dout[i].z += rsd[i].z;
        dout[i].w += rsd[i].w;
    }
}

template <>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    half *residual,
    half *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        dout[i] = __hadd2(dout[i], rsd[i]);
    } // addresidual
}

/*
在context_decoder阶段: num_tokens   在self_decoder阶段: batch_size
1. context_decoder 阶段：num_tokens -> 全量推理, 输入是整个句子, 帮助模型建立对输入的理解, 生成输出时, 利用先前的上下文信息
    num_tokens 通常是指 序列中 token 的数量, 也就是输入序列的长度
    例如, 在机器翻译或文本生成任务中, num_tokens 代表输入文本的长度, 可能是一个固定值或可变值
    在 context_decoder 阶段, 模型在处理输入时是基于上下文的
    这意味着在这个阶段, 模型处理的是多个 token, 而这些 token 的数量(num_tokens)决定了模型需要对多少个 token 执行操作
    具体来说, 这个阶段处理的是 输入序列的 token 数量, 可能是编码器输出的上下文表示
    eg: 一个输入句子 "Hello world!", 其中有 3 个 token(比如 Hello, world, 和 !), 在这个阶段, num_tokens = 3
        三个输入句子 "Hello world"(假设是 2 个 tokens） "How are you?"(假设是 4 个 tokens) "This is Llama2."(假设是 5 个 tokens)
        num_tokens: 11
2. self_decoder 阶段: batch_size -> 增量推理, 每次输出一个token, 输入是先前的句子 + 新生成的token
    batch_size 是指在一次计算中, 输入数据中样本的数量 
    在 self_decoder 阶段, 模型可能在做自回归生成(例如文本生成或翻译), 这个阶段通常涉及 多个序列 的同时处理 
    在这种情况下, batch_size 代表的是你一次性输入给模型的 样本数量
    每个样本的 token 数量通常是相同的(但也有例外, 比如变长序列处理时), 因此批量中的每个样本(序列)通常会有相同的 num_tokens 数量 
    eg: 一个输入句子 "Hello world!", batch_size: 1
        三个输入句子 "Hello world" "How are you?" "This is Llama2.", batch_size: 3
*/
template<typename T>
void launchAddResidual(             // 在context_decoder阶段: num_tokens   在self_decoder阶段: batch_size
    TensorWrapper<T> *residual,     // residual shape: [num_tokens, hidden_units]
    TensorWrapper<T> *decoder_out,  // decoder_out: [num_tokens, hidden_units]
    bool is_print) {
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    dim3 grid(num_tokens);
    dim3 block(256);
    AddResidual<<<grid, block>>>(residual->data, decoder_out->data, 
        num_tokens, hidden_units);

}

template void launchAddResidual(
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out,
    bool is_print);

template void launchAddResidual(
    TensorWrapper<half> *residual,
    TensorWrapper<half> *decoder_out,
    bool is_print);