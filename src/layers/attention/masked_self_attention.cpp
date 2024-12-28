// 与context_attention相比, 为什么没有padding和removePadding?
//  由于self_decoder, 即token_generate_phase是一个自回归的过程,
//  每次都是输入一个token到self_decoder中, 然后生成一个token, 循环上述过程
//  -> token_num: 1 -> 无需padding or removePadding

// masked self attention: 
//  fusedQkvGemm
//  AddbiasAndRope -----|
//  CnocatPastKVcache   |
//  Broadcast           |
//  Qk gemm             | -> FusedMaskedSelfAttention
//  Scale               |
//  Softmax             |
//  Qk*v gemm      -----|
//  Output linear

// input: [num_tokens] -> input_embedding: [num_tokens, hidden_size]
//                              |
//                              -> cal_paddingoffset: [bs, max_num_tokens, hidden_size]
//                              |
//                              -> build_casual_mask: mask: [bs, max_num_tokens, max_num_tokens]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_size] -> fusedQkvGemm: * [hidden_size, hidden_size] -> [num_tokens, hidden_size]
//                              -> AddbiasAndPaddingAndRope: [max_num_tokens, hidden_size] -> [bs, q_head_num, max_q_len, head_size]  ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            |                                       |
//                                            |                                       -> [bs, kv_head_num, max_q_len, head_size] ->
//                                            -> ConcatPastKVcache: [num_layers, bs, kv_head_num, max_seq_len(8192), head_size]
//                                            |     cache的内容: [bs, kv_head_num, seqlen[history_len : history_len + max_q_len], head_size]
//                                            |
//                                            -> Broadcast: kv: [bs, q_head_num, max_k_len, head_size]
//								-> Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
//                              -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len, hidden_size](bs*seq_len = num_tokens)
//                                  -> [num_tokens, hidden_size]
//                              -> FusedAddbiasResidual: [num_tokens, hidden_size]

#include <math.h>
#include "src/utils/debug_utils.h"
#include "src/layers/attention/masked_self_attention.h"

template<typename T>
LLaMaSelfAttentionLayer<T>::LLaMaSelfAttentionLayer(
	int head_num, int kv_head_num, int head_size,
	LLaMaAttentionStaticParams attn_params,
	cudaStream_t stream, cublasWrapper* cublas_wrapper,
	BaseAllocator* allocator):
	head_num(head_num), kv_head_num(kv_head_num),
	head_size(head_size), stream(stream),
	cublas_wrapper(cublas_wrapper), allocator(allocator),
	hidden_units(head_num * head_size),
	attn_static_params(attn_params),
	q_head_per_kv(head_num / kv_head_num),
	scale(float(1 / sqrt(head_size))) { }

template<typename T>
void LLaMaSelfAttentionLayer<T>::allocForForward(
	LLaMaAttentionDynParams &params) {
	DataType type = getTensorType<T>(); 
	const int batch_size = params.batch_size;
	const int num_tokens = params.num_tokens;
	const int max_q_len = params.max_q_len;
	const int max_k_len = params.max_k_len;
	const int qkv_head_num = head_num + 2 * kv_head_num;

	// build tnesorwrapper
	qkv_buf = new TensorWrapper<T>(Device::GPU, type,
		{batch_size, qkv_head_num, head_size});
	mha_output = new TensorWrapper<T>(Device::GPU, type,
		{batch_size, hidden_units});

	// buffer allocation
	qkv_buf->data = allocator->Malloc(qkv_buf->data,
		sizeof(T) * batch_size * qkv_head_num * head_size, false);
	mha_output->data = allocator->Malloc(mha_output->data,
		sizeof(T) * batch_size * hidden_units, false);
}

template<typename T>
void LLaMaSelfAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(mha_output->data);
    DeviceSyncAndCheckCudaError();
}

template<typename T>
void LLaMaSelfAttentionLayer<T>::forward(
	TensorMap& inputs, TensorMap& outputs,
	LLaMaAttentionWeights<T>& weights,
	LLaMaAttentionDynParams& params) {
	// alloc buffer
	allocForForward(params);

	// 1. qkv_linear
	// int hidden_units = (head_num + 2 * kv_head_num) * head_size;
	// int q_hidden_units = head_num * head_size;
	// [bs, q_hidden_units] *
	//	[q_hidden_units, hidden_units]
	Tensor *attention_input = inputs["attention_input"];
	launchLinearGemm(attention_input->as<T>(), weights.qkv,
		qkv_buf, cublas_wrapper);
	DeviceSyncAndCheckCudaError();

	// 2. RoPE
	// qkv_buf: [bs, hidden_units] ->
	//	[bs, qkv_head_num, head_size]
	//	bs: 不同句子, self_decoder默认为1
	Tensor *step = inputs["step"];
	LLaMaAttentionStaticParams attn_static_params = GetAttnStaticParams();
	launchRoPE(qkv_buf, step->as<int>(), attn_static_params);
	DeviceSyncAndCheckCudaError();

	// 3. FusedMaskedSelfAttention
	// [layer_id, bs, kv_head_num, head_size]
	Tensor *layer_id = inputs["layer_id"];
	Tensor *finished = inputs["finished"];
	Tensor *k_cache = outputs["all_k_cache"];
	Tensor *v_cache = outputs["all_v_cache"];
	Tensor *attention_output = outputs["attention_output"];
	launchDecoderMaskedMHA(qkv_buf, weights.qkv, layer_id->as<int>(),
		k_cache->as<T>(), v_cache->as<T>(), finished->as<bool>(), 
		step->as<int>(), mha_output, attn_static_params);
	DeviceSyncAndCheckCudaError();

	// 4. attention output linear
	launchLinearGemm(mha_output, weights.output,
		attention_output->as<T>(), cublas_wrapper, false, true);
	DeviceSyncAndCheckCudaError();

	this->freeBuf();
}

template class LLaMaSelfAttentionLayer<float>;