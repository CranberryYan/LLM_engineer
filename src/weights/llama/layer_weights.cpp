// 1. Embedding
// 2. RMSnorm
// 3. Qkbgemm
// 4. Output Linear
// 5. FusedAddbiasResidualandRMSnorm
// 6. Gate
// 7. Up
// 8. Down
// 9. Lmhead(Linear)

#include "src/utils/macro.h"
#include "src/weights/llama/layer_weights.h"

template <typename T>
LLaMALayerWeight<T>::LLaMALayerWeight(
  int head_num, int kv_head_num, int head_size,
  int inter_size, WeightType weight_type,
  bool attn_bias) {
  // RMSNorm: 均为hidden_units
  GPUMalloc(&attn_norm_weight.gamma, hidden_units);
  GPUMalloc(&ffn_norm_weight.gamma, hidden_units);

  // attention_weights
  self_attn_weight.qkv.wtype = weight_type;
  self_attn_weight.qkv.shape = {hidden_units,
    (head_num + 2 * kv_head_num) * head_size};
  GPUMalloc(&self_attn_weight.qkv.data,
    hidden_units * (head_num + 2 * kv_head_num) * head_size);
  self_attn_weight.output.wtype = weight_type;
  self_attn_weight.output.shape = {hidden_units, hidden_units};
  GPUMalloc(&self_attn_weight.output.data, hidden_units * hidden_units);

  if (attn_bias) {
  GPUMalloc(&self_attn_weight.qkv.bias,
    (head_num + 2 * kv_head_num) * head_size);
  GPUMalloc(&self_attn_weight.output.bias, hidden_units);
  }

  // ffn_weights
  ffn_weight.gate.wtype = weight_type;
  ffn_weight.up.wtype = weight_type;
  ffn_weight.down.wtype = weight_type;
  ffn_weight.gate.shape = {hidden_units, inter_size};
  ffn_weight.up.shape = {hidden_units, inter_size};
  ffn_weight.down.shape = {inter_size, hidden_units};
  GPUMalloc(&ffn_weight.gate.data, hidden_units * inter_size);
  GPUMalloc(&ffn_weight.up.data, hidden_units * inter_size);
  GPUMalloc(&ffn_weight.down.data, hidden_units * inter_size);
}

// load weights from HF model file
template <typename T>
void LLaMALayerWeight<T>::loadWeights(std::string weight_path,
  WeightType weight_type) {

}

// load weights from our self_defined dummy weights,
//  used to test preformance
template <typename T>
void LLaMALayerWeight<T>::loadWeights() {
    T *d_dummy_attn_norm_weight;
    T *d_dummy_ffn_norm_weight;
    T *d_dummy_qkv_weights;
    //T *d_dummy_qkv_bias;
    T *d_dummy_output_weights;
    T *d_dummy_output_bias;
    T *d_dummy_ffn_down;
    T *d_dummy_ffn_down_bias;
    T *d_dummy_ffn_gate_up;
    // T *d_dummy_ffn_up;
    CHECK(cudaMalloc((void**)&d_dummy_attn_norm_weight,
      sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_norm_weight,
      sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_qkv_weights,
      sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    // CHECK(cudaMalloc((void**)&d_dummy_qkv_bias,
    //  sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
    CHECK(cudaMalloc((void**)&d_dummy_output_weights,
      sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_output_bias,
      sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down,
      sizeof(T) * hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down_bias,
      sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_gate_up,
      sizeof(T) * hidden_units * 2 * inter_size));
    // CHECK(cudaMalloc(&d_dummy_ffn_up,
    //  sizeof(T) * hidden_units * inter_size));

    T *h_dummy_attn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T *h_dummy_ffn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T *h_dummy_qkv_weights = (T*)malloc(
      sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size);
    // T *h_dummy_qkv_bias = (T*)malloc(
    //  sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    T *h_dummy_output_weights = (T*)malloc(
      sizeof(T) * hidden_units * hidden_units);
    T *h_dummy_output_bias = (T*)malloc(sizeof(T) * hidden_units);
    T *h_dummy_ffn_down = (T*)malloc(sizeof(T) * hidden_units * inter_size);
    T *h_dummy_ffn_down_bias = (T*)malloc(sizeof(T) * hidden_units);
    T *h_dummy_ffn_gate_up = (T*)malloc(
      sizeof(T) * hidden_units * 2 * inter_size);
    // T *h_dummy_ffn_up = (T*)malloc(sizeof(T) * hidden_units * inter_size);

    for (int i = 0; i < hidden_units; i++){
        h_dummy_attn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_output_bias[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_down_bias[i] = (T)(rand() % 100 / (float)100000);
    }
    //for (int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++) {
    //    h_dummy_qkv_bias[i] = (T)(rand() % 100 / (float)100000);
    //}
    for (int i = 0; i < hidden_units * inter_size; i++) {
        h_dummy_ffn_down[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * 2 * inter_size; i++) {   
        h_dummy_ffn_gate_up[i] = (T)(rand() % 100 / (float)100000);
        // h_dummy_ffn_up[i] = (T)1.0f;
    }
    for (int i = 0; i < hidden_units * hidden_units; i++) {
        h_dummy_output_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * (head_num + 2 * kv_head_num) * head_size;
        i++) {
        h_dummy_qkv_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight,
      sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight,
      sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights,
      sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size,
      cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_dummy_qkv_bias, h_dummy_qkv_bias,
    //  sizeof(T) * (head_num + 2 * kv_head_num) * head_size,
    //  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights,
      sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias,
      sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down,
      sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias,
      sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up,
      sizeof(T) * hidden_units * 2 * inter_size, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_dummy_ffn_up, h_dummy_ffn_up,
    //  sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    // before kernel launch, the ptr is always void*,
    //  when luanching kernel, ptr type will be cast to float *or T*
    attn_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attn_weight.qkv.data = d_dummy_qkv_weights;
    self_attn_weight.qkv.bias = nullptr;
    self_attn_weight.output.data = d_dummy_output_weights;
    self_attn_weight.output.bias = d_dummy_output_bias;
    ffn_weight.gateAndup.data = d_dummy_ffn_gate_up;
    //ffn_weight.up.data = d_dummy_ffn_up;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
}

template <typename T>
void freeWeights(BaseWeight<T> &weights) {
  GPUFree(weights.data);
  if (weights.bias != nullptr) {
    GPUFree(weights.bias);
  }
  weights.data = nullptr;
  weights.bias = nullptr;
}

template <typename T>
LLaMALayerWeight<T>::~LLaMALayerWeight() {
  // free norm
  GPUFree(attn_norm_weight.gamma);
  GPUFree(ffn_norm_weight.gamma);

  // free weights
  freeWeights(self_attn_weight.qkv);
  freeWeights(self_attn_weight.output);
  freeWeights(ffn_weight.gate);
  freeWeights(ffn_weight.up);
  freeWeights(ffn_weight.down);
}