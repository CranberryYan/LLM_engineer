#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct LLaMaAttentionWeights {
    BaseWeight<T> q;
    BaseWeight<T> k;
    BaseWeight<T> v;
    BaseWeight<T> qkv;
    BaseWeight<T> output;
};
