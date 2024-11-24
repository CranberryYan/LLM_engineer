// SwiGLU: 结合了Swish和GLU的激活函数
//  Swish: Swish(x) = x * Sigmoid(x)
//  GLU: GLU(a, b) = a * Sigmoid(b)
// SwiGLU: SwiGLU(x) = (Linear1(x) ✖ Swish(Linear2(x)))

#include <iostream>
#include "src/kernels/act_kernel.h"

// silu: silu(x) = x * Sigmoid(x)
// sigmoid: sigmoid(x) = 1 / (1 + expf(-x))
template<typename T>
__device__ __forceinline__ T silu(const T &in) {
    return 1 / (1.0f + expf(-in));
}

template<>
__device__ __forceinline__ half2 silu(const half2 &in) {
    return make_half2(__float2half(silu<float>((float)in.x)), __float2half(silu<float>((float)in.y)));
}

// input: [bs, 2, inter_size] gate_linear和up_linear
// output: [bs, inter_size]
// 第一个inter_size去做silu, 结果与第二个inter_size做mul
template<typename T>
__global__ void silu_and_mul_kernel(T *output, 
    const T *input, const int intermedia_size) {
    const int batch_idx = blockIdx.x;
    for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
        const T x = input[batch_idx * 2 * intermedia_size + idx];
        const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
        output[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
    }
}

template<>
__global__ void silu_and_mul_kernel<half>(half *out,
    const half *input, const int intermedia_size) {
    const int batch_idx = blockIdx.x;
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
        const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + idx]));
        const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]));
        *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu<Vec_t>(x), y);
    }
}

template<typename T>
void launchAct(TensorWrapper<T> *input, TensorWrapper<T> *output) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);

    silu_and_mul_kernel<T><<<grid, block>>>(output->data, input->data, intermedia_size);
}

template void launchAct(TensorWrapper<float> *input, TensorWrapper<float> *output);
template void launchAct(TensorWrapper<half> *input, TensorWrapper<half> *output);
