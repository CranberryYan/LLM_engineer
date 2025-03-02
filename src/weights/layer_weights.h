#pragma once
#include "src/utils/weight_utils.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"

template<typename T>
class LlamaLayerWeight {
public:
  LlamaLayerWeight() = delete;
  LlamaLayerWeight(int head_num,
                  int  kv_head_num,
                  int  head_size,
                  int  inter_size,
                  WeightType weight_type,
                  bool attn_bias);
  ~LlamaLayerWeight();
public:
  void loadWeights();
  void loadWeights(std::string weight_path, WeightType weight_type);
  LayerNormWeight<T> attn_norm_weight;
  LayerNormWeight<T> ffn_norm_weight;
  LLaMaAttentionWeights<T> self_attn_weight;
  LLaMAFFNWeights<T> ffn_weight;

private:
  const int head_num;
  const int  kv_head_num;
  const int head_size;
  const int hidden_units;
  const int inter_size;
  const int bit_size;
  const bool attn_bias;
  WeightType weight_type;
};
