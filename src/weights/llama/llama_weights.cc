// input: [num_tokens] -> input_embedding: [num_tokens, hidden_units](num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)
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
//								-> Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
//                              -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len, hidden_units](bs*seq_len = num_tokens)
//                                  -> [num_tokens, hidden_units]
//                              -> FusedAddbiasResidual: [num_tokens, hidden_units]

#include <iostream>
#include "src/weights/llama/llama_weights.h"

template<typename T>
LlamaWeight<T>::LlamaWeight(int head_num, int kv_head_num, int head_size,
    int inter_size, int vocab_size, int num_layer, bool attn_bias,
    WeightType weight_type) :
  hidden_units(head_num * head_size), inter_size(inter_size),
  vocab_size(vocab_size), num_layer(num_layer), weight_type(weight_type) {
  llama_layer_weight.reserve(num_layer);
  for (int i = 0; i < num_layer; ++i) {
    // 1. LLaMALayerWeight
    //  std::vector<LLaMALayerWeight<T>*> llama_layer_weight;
    //    LLaMALayerWeight(int head_num,
    //      int  kv_head_num, int  head_size,
    //      int  inter_size, WeightType weight_type,
    //      bool attn_bias);
    llama_layer_weight.push_back(new LLaMALayerWeight<T>(head_num, kv_head_num,
      head_size, inter_size, weight_type, attn_bias));
  }

  // 2. LayerNormWeight
  //  template<typename T>
  //  struct LayerNormWeight {
  //      T *gamma;
  //  };
  GPUMalloc(&out_rmsnorm_weight.gamma, hidden_units);

  // 3. EmbeddingWeight
  //  // BaseWeight中的成员已经足够表达EmbeddingWeight的成员 -> 仅继承
  //  template<typename T>
  //  struct EmbeddingWeight: public BaseWeight<T> {};
  GPUMalloc(&pre_decoder_embedding_weight.data, vocab_size * hidden_units);
  GPUMalloc(&post_decoder_embedding_weight.data, vocab_size * hidden_units);

  pre_decoder_embedding_weight.shape = {vocab_size, hidden_units};
  post_decoder_embedding_weight.shape = {vocab_size, hidden_units};
  pre_decoder_embedding_weight.type = weight_type;
  post_decoder_embedding_weight.type = weight_type;
}

// weight from HF is always half type, and if we want run fp32 inference,
//  we should convert half weight to fp32 weight in tools/weights_convert.py
// shape and data of embedding and LMHead weight downloaded form HF 
//  are transposed, so we should carefully declare shape here
template<typename T>
void LlamaWeight<T>::loadWeights(std::string weight_path) {
  // 32xLayer的weight
  for (int layer = 0; layer < num_layer; ++layer) {
    llama_layer_weight[layer]->loadWeights(weight_path + 
      "model.layers." + std::to_string(layer), weight_type);
  }

  // out_rmsnorm_weight
  // pre_decoder_embedding_weight
  // post_decoder_embedding_weight
  loadWeightFromBin<T, float>::internalFunc(out_rmsnorm_weight.gamma,
    {(size_t)hidden_units}, weight_path + "model.norm.weight.bin");
  loadWeightFromBin<T, float>::internalFunc(pre_decoder_embedding_weight.data,
    {(size_t)vocab_size, (size_t)hidden_units},
    weight_path + "model.embed_tokens.weight.bin");
  loadWeightFromBin<T, float>::internalFunc(post_decoder_embedding_weight.data,
    {(size_t)vocab_size, (size_t)hidden_units},
    weight_path + "lm_head.weight.bin");
}


