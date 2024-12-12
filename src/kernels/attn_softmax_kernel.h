#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T>* qk,
    TensorWrapper<T>* mask, TensorWrapper<T>* attn_score, float scale);