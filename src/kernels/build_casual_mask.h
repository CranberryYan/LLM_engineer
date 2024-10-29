# pragma once

# include <cuda.h>
# include <cuda_fp16.h>
# include <cuda_runtime.h>
# include "src/utils/macro.h"
# include "src/utils/tensor.h"


template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, 
        TensorWrapper<int>* q_lens, TensorWrapper<int>* k_lens);