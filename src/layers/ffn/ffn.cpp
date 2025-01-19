// input: [num_tokens] -> input_embedding: [num_tokens, hidden_units](num_tokens: bs * q_len, q_len: 单个句子中的token集合, bs: 句子)
//                              |
//                              -> cal_paddingoffset: [bs, max_q_len, hidden_units]
//                              |
//                              -> build_casual_mask: mask: [bs, max_q_len, max_k_len]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_units] -> fusedQkvGemm: * [hidden_units, hidden_units] -> [num_tokens, hidden_units]
//                              -> AddbiasAndPaddingAndRope: [max_num_tokens, hidden_units] -> [bs, q_head_num, max_q_len, head_size]  ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            -> ConcatPastKVcache: [num_layers, bs, kv_head_num, max_seq_len(8192), head_size]
//                                            |     cache的内容: [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
//                                            |
//                                            -> Broadcast: kv: [bs, q_head_num, max_k_len, head_size]
//															-> Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
//                              -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len, hidden_units](bs*seq_len = num_tokens)
//                                  -> [num_tokens, hidden_units]
//                              -> FusedAddbiasResidual: [num_tokens, hidden_units]

// input: [bs, seq_len, hidden_units](来自FusedAddbiasResidual, context_attention的输出)
// Gate_linear和up_linear公用一块buffer(SwiGLU_input_buf)
// output: [bs, seq_len, hidden_units](输入FusedAddbiasResidual)
#include <iostream>
#include "src/layers/ffn/ffn.h"
#include "src/utils/debug_utils.h"

template<typename T>
LLaMaFFNLayer<T>::LLaMaFFNLayer(
	int head_num, int head_size, int inter_size,
	cudaStream_t stream, cublasWrapper* cublas_wrapper,
	BaseAllocator* allocator) :
	head_num(head_num),
	head_size(head_size),
	inter_size(inter_size),
	stream(stream),
	cublas_wrapper(cublas_wrapper),
	allocator(allocator),
	hidden_units(head_num * head_size) { }

// for context_decoder
//	因为此时removePadding, [bs, seq_len] -> num_tokens
template<typename T>
void LLaMaFFNLayer<T>::allocForForward(
	LLaMaAttentionDynParams &params) {
	int num_tokens = params.num_tokens;
	DataType type = getTensorType<T>();
	SwiGLU_input = new TensorWrapper<T>(Device::GPU, type,
		{num_tokens, 2, inter_size});
	down_proj_input = new TensorWrapper<T>(Device::GPU, type,
		{num_tokens, inter_size});
	SwiGLU_input->data = allocator->Malloc(
		SwiGLU_input->data,
		sizeof(T) * num_tokens * 2 * inter_size, false);
	down_proj_input->data = allocator->Malloc(
		down_proj_input->data,
		sizeof(T) * num_tokens * inter_size, false);
}

// for self_decoder
//	没有padding, 因为此时的seq_len均为1(仅输出单个token)
template<typename T>
void LLaMaFFNLayer<T>::allocForForward(
	int batch_size){
	DataType type = getTensorType<T>();
	SwiGLU_input = new TensorWrapper<T>(Device::GPU, type,
		{batch_size, 2, inter_size});
	down_proj_input = new TensorWrapper<T>(Device::GPU, type,
		{batch_size, inter_size});
	SwiGLU_input->data = allocator->Malloc(
		SwiGLU_input->data,
		sizeof(T) * batch_size * 2 * inter_size, false);
	down_proj_input->data = allocator->Malloc(
		down_proj_input->data,
		sizeof(T) * batch_size * inter_size, false);
}

template<typename T>
void LLaMaFFNLayer<T>::freeBuf(){
    allocator->Free(SwiGLU_input->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(down_proj_input->data);
    DeviceSyncAndCheckCudaError();
}

template<typename T>
void LLaMaFFNLayer<T>::forward(TensorMap& inputs, TensorMap& outputs,
	LLaMAFFNWeights<T>& weights, LLaMaAttentionDynParams& params){
	if (params.num_tokens > 0) {
		allocForForward(params);
	} else {
		allocForForward(params.batch_size);
	}
	Tensor* ffn_input = inputs["ffn_input"];
	Tensor* ffn_output = outputs["ffn_output"];
	bool is_ctx = params.is_ctx;

	// 1.fusedGateUp proj
	std::cout << "=================== enter launchLinearGemm_1" << std::endl;
	launchLinearGemm(ffn_input->as<T>(),
		weights.gateAndup, SwiGLU_input, cublas_wrapper, false, true);
	DeviceSyncAndCheckCudaError();
	// single up proj linear, deprecated due to fuse gate and up into fusedGateAndup
	// launchLinearGemm(ffn_input->as<T>(), weights.up, SwiGLU_input, cublas_wrapper, false, false, true);

	// 2.swiGLU
	std::cout << "=================== enter launchAct" << std::endl;
	launchAct(SwiGLU_input, down_proj_input);// down_proj_input maybe can reuse swiglu_input buf, will validate it later
	DeviceSyncAndCheckCudaError();

	// 3.down proj
	std::cout << "=================== enter launchLinearGemm_2" << std::endl;
	launchLinearGemm(down_proj_input,
		weights.down, ffn_output->as<T>(), cublas_wrapper, false, true);
	DeviceSyncAndCheckCudaError();

	this->freeBuf();
};

template class LLaMaFFNLayer<float>;
