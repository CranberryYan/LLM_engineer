#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"
template <typename T>
void launchAddResidual(TensorWrapper<T> *residual, 
    TensorWrapper<T> *decoder_out, bool is_print=false);