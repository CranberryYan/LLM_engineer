#include <stdio.h>
#include "src/kernels/add_residual.h"


template<typename T>
__global__ void AddResidual(T *residual, T *decoder_out,
    int num_tokens, int hidden_units) {
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout[i].x += rsd[i].x;
        dout[i].y += rsd[i].y;
        dout[i].z += rsd[i].z;
        dout[i].w += rsd[i].w;
    }
}

template<typename T>
void launchAddResidual(             // 在context_decoder阶段: num_tokens   在self_decoder阶段: batch_size
    TensorWrapper<T> *residual,     // residual shape: [num_tokens, hidden_units]
    TensorWrapper<T> *decoder_out,  // decoder_out: [num_tokens, hidden_units]
    bool is_print) {
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    dim3 grid(batch_size);
    dim3 block(256);
    AddResidual<<<grid, block>>>(residual->data, decoder_out->data, 
        batch_size, hidden_units);

}
template void launchAddResidual(
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out,
    bool is_print);