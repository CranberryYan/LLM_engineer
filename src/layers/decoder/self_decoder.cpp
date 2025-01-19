#include <iostream>
#include "src/utils/macro.h"
#include "src/layers/decoder/self_decoder.h"

template <typename T>
void LlamaSelfDecoder<T>::allocForForward(LLaMaAttentionDynParams &dyn_params) {
  DataType type = getTensorType<T>(); 
  int batch_size = dyn_params.batch_size;
  decoder_residual = new TensorWrapper<T>(Device::GPU,
    type, {batch_size, hidden_units});
  decoder_residual->data = allocator->Malloc(decoder_residual->data,
    sizeof(T) * batch_size * hidden_units, false);
}

template <typename T>
void LlamaSelfDecoder<T>::freeBuf() {
  allocator->Free(decoder_residual->data);
  DeviceSyncAndCheckCudaError();
}

// 与context_decoder相比, 没有paddingoffset, causelmasks
template <typename T>
void LlamaSelfDecoder<T>::forward(TensorMap &input_tensors,
    const std::vector<LLaMALayerWeight<T> *> &layerWeights,
    TensorMap &output_tensors, LLaMaAttentionDynParams &dyn_params) {
    allocForForward(dyn_params);
    Tensor* decoder_input = input_tensors["decoder_input"];
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* step = input_tensors["step"];
    Tensor* finished = input_tensors["finished"];
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    Tensor* layer_id = input_tensors["layer_id"];
    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->as<int>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->as<bool>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

  TensorMap self_attn_inputs {
    {"attention_input", decoder_input},
    {"step", step},
    {"finished", finished},
    {"layer_id", layer_id}
  };
  TensorMap self_attn_outputs {
    {"attention_output", decoder_output},
    {"all_k_cache", all_k_cache},
    {"all_v_cache", all_v_cache}
  };
  for (int layer_id_ = 0; layer_id_ < 32; ++layer_id_) {
    if (layer_id_ > 0) {
      TensorWrapper<int>* layer =
        new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id_);
      self_attn_inputs.insert("layer_id", layer);
    }
    // 首尾相连, 本次输出作为下一次的输入
    decoder_input = self_attn_inputs["attention_input"];
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
    selfAttn->forward(self_attn_inputs, self_attn_outputs,
      layerWeights[layer_id_]->self_attn_weight, dyn_params);

    // void launchFusedAddBiasResidualRMSNorm(
    //  TensorWrapper<T> *residual, TensorWrapper<T> *decoder_in, // [num tokens, hidden_units]
    //  BaseWeight<T> &norm, T *scale,                            //RMSNorm weights
    //  float eps);
    launchFusedAddBiasResidualRMSNorm(decoder_residual, decoder_output->as<T>(),
      layerWeights[layer_id_]->self_attn_weight.output,
      layerWeights[layer_id_]->ffn_norm_weight.gamma,
      rmsnorm_eps);
    DeviceSyncAndCheckCudaError();

    TensorMap ffn_inputs{
      {"ffn_input", decoder_output}
    };
    TensorMap ffn_outputs{
      {"ffn_output", decoder_output}
    };
    ffn->forward(ffn_inputs, ffn_outputs,
      layerWeights[layer_id_]->ffn_weight, dyn_params);

    // void launchAddResidual(TensorWrapper<T> *residual, 
    //   TensorWrapper<T> *decoder_out, bool is_print=false);
    launchAddResidual(decoder_residual, decoder_output->as<T>(), false);
    DeviceSyncAndCheckCudaError();

    // 首尾相连, 本次输出作为下一次的输入
    self_attn_inputs.insert("attention_input", decoder_output);
  }
  freeBuf();
  DeviceSyncAndCheckCudaError();
}

template class LlamaSelfDecoder<float>;