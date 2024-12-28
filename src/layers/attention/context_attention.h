#pragma once
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/kernels/repeat_kv.h"
#include "src/kernels/cublas_utils.h"
#include "src/kernels/concat_past_kv.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/fused_transpose_and_remv_pad.h"

template<typename T>
class LLaMaContextAttentionLayer
{
public:
	LLaMaContextAttentionLayer(int head_num, int kv_head_size, int head_size,
		LLaMaAttentionStaticParams attn_params, cudaStream_t stream,
		cublasWrapper *cublas_wrapper, BaseAllocator *allocator);

public:
    void allocForForward(LLaMaAttentionDynParams& params);
    
		void freeBuf(); // free 中间buf
    
		void forward(TensorMap& inputs, TensorMap& outputs, 
			LLaMaAttentionWeights<T>& weights, 
			LLaMaAttentionDynParams& params, 
			LLaMaAttentionStaticParams& static_params);

	// 返回的是 attn_static_params 的引用,
	//	即调用此函数时不会复制 attn_static_params, 而是直接返回对原变量的访问
	LLaMaAttentionStaticParams& GetAttnStaticParams(){
			return attn_static_params;
	}

private:
	const int head_num;				// q_head_num
	const int head_size;
	const int hidden_units;
	const int q_head_per_kv; 	// GQA and MQA
	const int kv_head_num;
	float scale;							// sqrt(head_size)

	cudaStream_t stream;
	LLaMaAttentionStaticParams attn_static_params;
	BaseAllocator *allocator;
	cublasWrapper *cublas_wrapper;

	TensorWrapper<T> *qkv_buf_wo_pad = nullptr; // wo: without     
	TensorWrapper<T> *q_buf_w_pad = nullptr;		// w: with
	TensorWrapper<T> *k_buf_w_pad = nullptr;
	TensorWrapper<T> *v_buf_w_pad = nullptr;
	TensorWrapper<T> *k_cache_buf = nullptr;
	TensorWrapper<T> *v_cache_buf = nullptr;
	TensorWrapper<T> *qk_buf = nullptr;
	TensorWrapper<T> *qkv_buf_w_pad = nullptr;
	TensorWrapper<T> *qkv_buf_wo_pad_1 = nullptr;
};
