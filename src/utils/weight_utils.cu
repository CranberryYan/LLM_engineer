#include "src/utils/weight_utils.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size) {
	LLM_CHECK_WITH_INFO(size >= 0, "request cudaMalloc size" +
		std::to_string(size) + "< 0, which is invalid");
	CHECK(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

template<typename T>
void GPUFree(T *ptr){
	if (ptr != nullptr) {
		CHECK(cudaFree(ptr));
		ptr = nullptr;
	}
}

template 
void GPUMalloc(float** ptr, size_t size);
template
void GPUFree(float *ptr);

