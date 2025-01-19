#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "src/utils/macro.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size);

template<typename T>
void GPUFree(T *ptr);