#pragma once
#include "src/utils/weight_utils.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"

// 1. Embedding(不算layer Weights, 因为整个网络仅有一次Embedding, 算Llama Weights)
// 2. RMSnorm
// 3. Qkbgemm
// 4. Output Linear
// 5. FusedAddbiasResidualandRMSnorm
// 6. Gate
// 7. Up
// 8. Down
// 9. Lmhead(Linear)(同Embedding)
template<typename T>
class LLaMALayerWeight {
public:
  LLaMALayerWeight() = delete;
  LLaMALayerWeight(int head_num,
    int  kv_head_num, int  head_size,
    int  inter_size, WeightType weight_type,
    bool attn_bias);
  ~LLaMALayerWeight();
public:
  void loadWeights();
  void loadWeights(std::string weight_path, WeightType weight_type);
  LayerNormWeight<T> attn_norm_weight;
  LayerNormWeight<T> ffn_norm_weight;
  LLaMaAttentionWeights<T> self_attn_weight;
  LLaMAFFNWeights<T> ffn_weight;

private:
  const int head_num;
  const int kv_head_num;
  const int head_size;
  const int hidden_units;
  const int inter_size;
  const int bit_size;
  const bool attn_bias;
  WeightType weight_type;
};
