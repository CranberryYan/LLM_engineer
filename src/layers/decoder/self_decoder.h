#pragma once
#include "src/kernels/add_residual.h"
#include "src/kernels/rmsnorm_kernel.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/kernels/fused_decoder_self_attention.h"
#include "src/utils/tensor.h"
#include "src/layers/ffn/ffn.h"
#include "src/layers/attention/masked_self_attention.h"
#include "src/weights/llama/llama_weights.h"

template <typename T>
class LlamaSelfDecoder {
public:
  LlamaSelfDecoder(int head_num, int kv_head_num, int head_size,
    int inter_size, int num_layer,
    const LLaMaAttentionStaticParams &attn_params,
    float rmsnorm_eps, cudaStream_t stream, cublasWrapper *cublas_wrapper,
    BaseAllocator *allocator) : head_num(head_num), head_size(head_size),
    inter_size(inter_size), hidden_units(head_num * head_size),
    num_layer(num_layer), rmsnorm_eps(rmsnorm_eps),
    data_type(getTensorType<float>()), stream(stream),
    cublas_wrapper(cublas_wrapper), allocator(allocator) {
    selfAttn = new LLaMaSelfAttentionLayer<T>(head_num,
                      kv_head_num, head_size, attn_params, stream,
                      cublas_wrapper, allocator);

    ffn = new LLaMaFFNLayer<T>(head_num, head_size, inter_size,
                stream, cublas_wrapper, allocator);
  }

public:
  void allocForForward(LLaMaAttentionDynParams &dyn_params);
  void freeBuf();
  void forward(TensorMap &input_tensors,
    const std::vector<LLaMALayerWeight<T> *> &layerWeights,
    TensorMap &output_tensors, LLaMaAttentionDynParams &dyn_params);

private:
  int head_num;
  int kv_head_num;
  int head_size;
  int inter_size;
  int num_layer;
  int hidden_units;
  float rmsnorm_eps;

  cudaStream_t stream;
  cublasWrapper *cublas_wrapper;
  BaseAllocator *allocator;

  TensorWrapper<T> *decoder_residual;

  LLaMaSelfAttentionLayer<T> *selfAttn;
  LLaMaFFNLayer<T> *ffn;
  DataType data_type;
};
