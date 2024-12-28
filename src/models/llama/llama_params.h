#pragma once
struct LLaMaAttentionStaticParams {
    int		rotary_embedding_dim;
    float	rotary_embedding_base;
    int		max_position_embeddings;
    bool	use_dynamic_ntk;
};

struct LLaMaAttentionDynParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
    bool is_ctx = false;
};
