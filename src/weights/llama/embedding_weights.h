# pragma once
# include "src/weights/base_weights.h"

// BaseWeight中的成员已经足够表达EmbeddingWeight的成员 -> 仅继承
template<typename T>
struct EmbeddingWeight: public BaseWeight<T> {};