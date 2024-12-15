#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"
#include "src/weights/llama/norm_weights.h"

template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_out,
	LayerNormWeight<T> &attn_norm_weight, float eps);