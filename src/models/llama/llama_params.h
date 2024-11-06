#pragma once
struct LLaMAAttentionStaticParams {
    int		rotary_embedding_dim;
    float	rotary_embedding_base;
    int		max_position_embeddings;
    bool	use_dynamic_ntk;
};
