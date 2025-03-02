// context_decoder
//  1. RMSNorm input
//    input embedding
//    CalPaddingOffset
//    BuildCausalMask
//  2. Context Decoder Layer x32
//    Context Attention
//    FFN
#pragma once
#include "src/kernels/add_residual.h"
#include "src/kernels/rmsnorm_kernel.h"
#include "src/kernels/build_casual_mask.h"
#include "src/kernels/cal_paddingoffset.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/layers/ffn/ffn.h"
#include "src/layers/attention/context_attention.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"

template <typename T>
class LlamaContextDecoder {
public:
  LlamaContextDecoder(int head_num,
    int kv_head_num, int head_size,
    int inter_size, int num_layer,
    const LLaMaAttentionStaticParams &attn_params,
    float rmsnorm_eps, cudaStream_t stream,
    cublasWrapper *cublas_wrapper, BaseAllocator *allocator) :
    head_num(head_num), head_size(head_size), inter_size(inter_size),
    hidden_units(head_num * head_size), num_layer(num_layer),
    rmsnorm_eps(rmsnorm_eps), data_type(getTensorType<T>()),
    stream(stream), cublas_wrapper(cublas_wrapper), allocator(allocator) {
    ctxAttn = new LLaMaContextAttentionLayer<T>(head_num, kv_head_num,
                    head_size, attn_params, stream, cublas_wrapper, allocator);
    ffn = new LLaMaFFNLayer<T>(head_num, head_size, inter_size,
                stream, cublas_wrapper, allocator);
};
  void allocForForward(LLaMaAttentionDynParams &dyn_params);
  void freeBuf();
  void forward(TensorMap &inputs,
    const std::vector<LlamaLayerWeight<T>*> &layerWeights,
    TensorMap &output, LLaMaAttentionDynParams& dyn_params);

private:
  int head_num;
  int kv_head_num;
  int head_size;
  int inter_size;
  int num_layer;
  int hidden_units;
  float rmsnorm_eps;
  TensorWrapper<T> *attention_mask;
  TensorWrapper<int> *padding_offset;
  TensorWrapper<int> *cum_seqlens;
  TensorWrapper<T> *decoder_residual;
  cudaStream_t stream;
  cublasWrapper *cublas_wrapper;
  BaseAllocator *allocator;

  LLaMaFFNLayer<T> *ffn;
  LLaMaContextAttentionLayer<T> *ctxAttn;
  DataType data_type;
};
