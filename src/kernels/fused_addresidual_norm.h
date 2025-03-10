#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/weights/base_weights.h"
#include "src/weights/llama/norm_weights.h"
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

// residual.shape = [num tokens, hidden_units]
template<typename T>
void launchFusedAddBiasResidualRMSNorm(
    TensorWrapper<T> *residual, TensorWrapper<T> *decoder_in, // [num tokens, hidden_units]
    BaseWeight<T> &norm, T *scale,                            //RMSNorm weights
    float eps);