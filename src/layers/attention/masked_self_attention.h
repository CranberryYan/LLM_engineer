// masked self attention: 
//  fusedQkvGemm
//  AddbiasAndRope
//  CnocatPastKVcache  -|
//  Broadcast           |
//  Qk gemm             | -> FusedMaskedSelfAttention
//  Scale               |
//  Softmax             |
//  Qk*v gemm      -----|
//  Output linear

#pragma once
#include "src/utils/macro.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"                         // 1st/4th kernel of masked self attention, qkv gemm
#include "src/kernels/qkv_bias_and_RoPE.h"				// 2ed RoPE
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/fused_decoder_self_attention.h"   // 3rd kernel 

// 这里面的数据成员都是只存在于attention_layer，而不像finished，seq_lengths这种贯穿整个过程
template<typename T>
class LLaMaSelfAttentionLayer {
public:
	LLaMaSelfAttentionLayer(int head_num,
		int kv_head_num, int head_size,
		LLaMaAttentionStaticParams attn_params,
		cudaStream_t stream, cublasWrapper* cublas_wrapper,
		BaseAllocator* allocator);

	LLaMaAttentionStaticParams& GetAttnStaticParams(){
			return attn_static_params;
	}
public:
	void allocForForward(LLaMaAttentionDynParams& params);
	void freeBuf();
	void forward(TensorMap& inputs, TensorMap& outputs,
		LLaMaAttentionWeights<T>& weights, LLaMaAttentionDynParams& params);

private:
	// this params are shared across all LLMs
	float scale;
	const int head_num;
	const int head_size;
	const int hidden_units;
	const int q_head_per_kv; //for GQA or MQA
	const int kv_head_num;


	// this params are only saw in llama and are unchanged 
	LLaMaAttentionStaticParams attn_static_params;
	cudaStream_t stream;
	BaseAllocator* allocator;

	// for linear and gemm
	cublasWrapper* cublas_wrapper;

	// intermedia buffer
	TensorWrapper<T> *qkv_buf    = nullptr; // for qkv linear output and rope input/output
	TensorWrapper<T> *mha_output = nullptr; // mha output, then invoke a linear to attention output
};
