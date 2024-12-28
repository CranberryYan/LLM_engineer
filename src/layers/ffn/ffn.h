#pragma once
#include "src/kernels/linear.h"
#include "src/kernels/act_kernel.h"
#include "src/kernels/cublas_utils.h"
#include "src/utils/macro.h"
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/weights/llama/attention_weights.h"

template<typename T>
class LLaMaFFNLayer {
public:
	LLaMaFFNLayer(int head_num, int head_size,
		int inter_size, cudaStream_t stream,
		cublasWrapper *cublas_wrapper,
		BaseAllocator *allocator);
public:
	void allocForForward(LLaMaAttentionDynParams &params);
	void allocForForward(int batch_size);
	void freeBuf();
	void forward(TensorMap &inputs, TensorMap &outpus,
		LLaMAFFNWeights<T> &weights,
		LLaMaAttentionDynParams &params);

private:
	const int head_num;
	const int head_size;
	const int inter_size;
	const int hidden_units;

	cudaStream_t stream;
	BaseAllocator *allocator;
	cublasWrapper *cublas_wrapper;
	// [2, num_tokens, intersize]
	TensorWrapper<T> *SwiGLU_input = nullptr;
	// [num_tokens, intersize]
	TensorWrapper<T> *down_proj_input = nullptr;
};
