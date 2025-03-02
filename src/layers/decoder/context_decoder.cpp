// input: [num_tokens] -> input_embedding: [num_tokens, hidden_units](num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)(num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)
//                              |
//                              -> cal_paddingoffset: [bs, max_q_len, hidden_units]
//                              |
//                              -> build_casual_mask: mask: [bs, max_q_len, max_k_len]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_units] -> fusedQkvGemm: * [hidden_units, hidden_units] -> [num_tokens, hidden_units]
//                              -> AddbiasAndPaddingAndRope: [max_num_tokens, hidden_units] -> [bs, q_head_num, max_q_len, head_size]  ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            -> ConcatPastKVcache: [num_layers, bs, kv_head_num, max_seq_len(8192), head_size]
//                                            |     cache的内容: [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
//                                            |
//                                            -> Broadcast: kv: [bs, q_head_num, max_k_len, head_size]
//										 -> Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
//                     -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len, hidden_units](bs*seq_len = num_tokens)
//                      	-> [num_tokens, hidden_units]
//                     -> FusedAddbiasResidual: [num_tokens, hidden_units]
#include <iostream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/decoder/context_decoder.h"

template <typename T>
void LlamaContextDecoder<T>::allocForForward(
  LLaMaAttentionDynParams &dyn_params) {
  int num_tokens = dyn_params.num_tokens;
  int batch_size = dyn_params.batch_size;
  int max_q_len = dyn_params.max_q_len;
  int max_k_len = dyn_params.max_k_len;
  DataType type = getTensorType<T>(); 
  DataType type_int = getTensorType<int>();

  // FusedAddbiasResidual: [num_tokens, hidden_units]
  decoder_residual = new TensorWrapper<T>(Device::GPU, type,
    {num_tokens, hidden_units});

  // build_casual_mask: mask: [bs, max_q_len, max_k_len]
  attention_mask = new TensorWrapper<T>(Device::GPU, type,
    {batch_size, max_q_len, max_k_len});

  // cal_paddingoffset: [bs, max_q_len, hidden_units]
  padding_offset = new TensorWrapper<int>(Device::GPU, type_int,
    {batch_size, max_q_len});
  cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int,
    {batch_size + 1});

  // data
  decoder_residual->data = allocator->Malloc(decoder_residual->data,
    sizeof(T) * num_tokens * hidden_units, false);
  attention_mask->data = allocator->Malloc(attention_mask->data,
    sizeof(T) * batch_size * max_q_len * max_k_len, false);
  padding_offset->data = allocator->Malloc(padding_offset->data,
    sizeof(int) * batch_size * max_q_len, false);
  cum_seqlens->data = allocator->Malloc(cum_seqlens->data,
    sizeof(int) * (batch_size + 1), false);
}

template <typename T>
void LlamaContextDecoder<T>::freeBuf() {
  allocator->Free(attention_mask->data);
  DeviceSyncAndCheckCudaError();
  allocator->Free(padding_offset->data);
  DeviceSyncAndCheckCudaError();
  allocator->Free(cum_seqlens->data);
  DeviceSyncAndCheckCudaError();
}

template <typename T>
void LlamaContextDecoder<T>::forward(TensorMap &inputs,
  const std::vector<LlamaLayerWeight<T>*> &layerWeights,
  TensorMap &output, LLaMaAttentionDynParams& dyn_params) {
  allocForForward(dyn_params);
  Tensor* seq_lens = inputs["input_length"];

  // 1. calculate padding offset
  // void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,
  // 			TensorWrapper<int>* cum_seqlens,   // 累计句子长度
  // 			TensorWrapper<int>* input_lengths) // 实际输入长度
  // shape:
  //  seq_lengths:[bs]
  //  output cum_seqlens:[bs + 1], first ele is 0
  //  output padding_offset:[bs * max_q_len]
  launchCalPaddingoffset(padding_offset, cum_seqlens, seq_lens->as<int>());
  DeviceSyncAndCheckCudaError();

  // 2. build causal mask
  // void launchBuildCausalMasks(TensorWrapper<T> *mask,     
  //      TensorWrapper<int>* q_lens, TensorWrapper<int>* k_lens)
  Tensor* context_lens = inputs["context_length"];
  launchBuildCausalMasks(attention_mask,
    seq_lens->as<int>(), context_lens->as<int>());
  DeviceSyncAndCheckCudaError();

  // 3.context_attn
  // 32x context_docoder_layer
  Tensor* layer_id = inputs["layer_id"];
  Tensor* history_length = inputs["history_length"];
  Tensor* decoder_input = inputs["decoder_input"];
  Tensor* decoder_context = inputs["decoder_context"];
  Tensor* decoder_output = output["decoder_output"];
  Tensor* all_k_cache = output["all_k_cache"];
  Tensor* all_v_cache = output["all_v_cache"];
  LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr,
    "the data ptr of tensor inserted into TensorMap is nullptr!");
  LLM_CHECK_WITH_INFO(history_length->as<int>()->data != nullptr,
    "the data ptr of tensor inserted into TensorMap is nullptr!");
  TensorMap ctx_attn_inputs {
    {"attention_input", decoder_input},
    {"padding_offset", padding_offset},
    {"history_length", history_length},
    {"input_length", seq_lens},
    {"context_length", context_lens},
    {"attention_mask", attention_mask},
    {"layer_id", layer_id}
  };
  TensorMap ctx_attn_outputs {
    {"attention_output", decoder_output},
    {"all_k_cache", all_k_cache},
    {"all_v_cache", all_v_cache}
  };
  TensorMap ffn_inputs{
    {"ffn_input", decoder_output}
  };
  TensorMap ffn_outputs{
    {"ffn_output", decoder_output}
  };
  DataType type_int = getTensorType<int>();
  DataType type = getTensorType<T>();
  for (int layer_id_ = 0; layer_id_ < 32; ++layer_id_) {
    if (layer_id_ > 0) {
      TensorWrapper<int>* layer =
        new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id_);
      ctx_attn_inputs.insert("layer_id", layer);
    }
    // 首尾相连, 本次输出作为下一次的输入
    decoder_input = ctx_attn_inputs["attention_input"];
    // void launchRMSNorm(TensorWrapper<T> *decoder_in,
    // 	TensorWrapper<T> *decoder_residual,
    // 	LayerNormWeight<T> &attn_norm_weight, float eps)
    launchRMSNorm(decoder_input->as<T>(), decoder_residual,
      layerWeights[layer_id_]->attn_norm_weight, rmsnorm_eps);
    DeviceSyncAndCheckCudaError();

    // void forward(TensorMap &inputs, TensorMap &outputs,
    //  LLaMaAttentionWeights<T> &weights,
    //  LLaMaAttentionDynParams &params,
    //  LLaMaAttentionStaticParams &static_params)
    ctxAttn->forward(ctx_attn_inputs, ctx_attn_outputs,
      layerWeights[layer_id_]->self_attn_weight, dyn_params,
      ctxAttn->GetAttnStaticParams());

    // void launchFusedAddBiasResidualRMSNorm(
    //  TensorWrapper<T> *residual, TensorWrapper<T> *decoder_in, // [num tokens, hidden_units]
    //  BaseWeight<T> &norm, T *scale,                            //RMSNorm weights
    //  float eps);
    launchFusedAddBiasResidualRMSNorm(decoder_residual, decoder_output->as<T>(),
      layerWeights[layer_id_]->self_attn_weight.output,
      layerWeights[layer_id_]->ffn_norm_weight.gamma,
      rmsnorm_eps);
    DeviceSyncAndCheckCudaError();

    ffn->forward(ffn_inputs, ffn_outputs,
      layerWeights[layer_id_]->ffn_weight, dyn_params);

    // void launchAddResidual(TensorWrapper<T> *residual, 
    //   TensorWrapper<T> *decoder_out, bool is_print=false);
    launchAddResidual(decoder_residual, decoder_output->as<T>(), false);
    DeviceSyncAndCheckCudaError();

    // 首尾相连, 本次输出作为下一次的输入
    ctx_attn_inputs.insert("attention_input", decoder_output);
  }
  freeBuf();
  DeviceSyncAndCheckCudaError();
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;
