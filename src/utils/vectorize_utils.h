# pragma once
# include <cuda.h>
# include <cuda_fp16.h>


// 向量化load和store时, 会用到
// FP32   FP16
template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0; // 为了后续的模板特化
};

template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

template<>
struct Vec<half> {
    using Type = half2;
    static constexpr int size = 2;
};
