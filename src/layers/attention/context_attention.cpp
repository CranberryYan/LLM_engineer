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
//                              -> RemovePadding: [bs, q_head_num, seq_len, head_size] -> [bs, seq_len, q_head_num, head_size] -> [bs, seq_len(num_tokens), hidden_size]
//                              -> FusedAddbiasResidual: [bs, seq_len, hidden_size]

// fusedQkvGemm -> AddbiasAndPaddingAndRope -> ConcatPastKVcache -> broadcast(GQA or MQA)(Llama2: MHA, 无broadcast)
//  -> Qk gemm -> FusedMaskAndScaleSoftmax
//  -> Qk * v gemm -> RemovingPadding -> output linear

// whats the diff across these 3 max len:
// max_seq_len is the max kv len considering context, ep. multiple epochs chat
// max_q_len is the current max q len after padding in this batch
// all kv cache is max seq len to save all kv cache in all epochs, 
//	but in context attention, 
//	all kv cache should be broadcast to adapt q as kv cache buf 
//	whose shape is max k len
// max k len is the max context len in cur batch  
// void flashAttn();
// seq_len q_len k_len 的区别:
//  max_seq_len: KV Cache 的全局最大长度, 包含了模型从头到当前推理时积累的所有上下文(例如跨多轮对话的 KV Cache);
//  max_q_len: 当前batch中, 经过 padding 后 Query 的最大长度(padding标准就是当前batch中最长的句子)
//  max_k_len: 当前batch中, 经过上下文的最大长度(包括 Key 缓存的实际范围)
//  在多轮生成或推理中:
//      上下文长度：指 Query 在当前批次中可以参考的历史序列(Key 缓存)的最大长度;
//      这个长度取决于以下几个因素: 
//          历史序列的长度: 之前保存的 Key-Value 缓存的长度;
//          通常取决于有多少个 token(即历史上下文中存储的 Key-Value 对的数量);
//          当前 Query 的需求: 每个 Query 需要对哪些 Key-Value 进行注意力操作;
//          模型限制：模型可能设置了最大上下文长度, 比如 max_seq_len;
//          因此, 上下文的最大长度可以理解为当前 Query 能够"看到"的 Key 的最大范围;

#include <math.h>
#include "src/utils/macro.h"
#include "src/layers/attention/context_attention.h"

template<typename T>
LLaMaContextAttentionLayer<T>::LLaMaContextAttentionLayer(int head_num, int kv_head_num, int head_size,
		LLaMAAttentionStaticParams attn_params, cudaStream_t stream,
		cublasWrapper *cublas_wrapper, BaseAllocator *allocator) :
    head_num(head_num), kv_head_num(kv_head_num), head_size(head_size),
    stream(stream), cublas_wrapper(cublas_wrapper), allocator(allocator), 
    hidden_units(head_num * head_size), attn_static_params(attn_params),
    q_head_per_kv(head_num / kv_head_num), scale(float(1 / sqrt(head_size))){ }

template<typename T>
void LLaMAContextAttentionLayer<T>::allocForForward(
	LLaMAAttentionDynParams& params) {
	DataType type = getTensorType<T>();
	int batch_size = params.batch_size;
	int num_tokens = params.num_tokens;
	int max_q_len = params.max_q_len;
	int max_k_len = params.max_k_len;
	const int qkv_head_num = head_num + 2 * kv_head_num;

	// qkv linear and bias rope
	// why here isn't max_k_len(all max_q_len)?
	//	cause the q/k/v is got by {bs, q_len, hiddenunits} * {hiddenunits, hiddenunits}
	//  -> AddbiasAndPaddingAndRope: [max_num_tokens, hidden_size]
	//																						 |
	//																						 -> [bs, q_head_num, max_q_len, head_size]
	//                                             |
	//                                             -> [bs, kv_head_num, max_q_len, head_size]
	//                                             |
	//                                             -> [bs, kv_head_num, max_q_len, head_size]
	qkv_buf_wo_pad = new TensorWrapper<T>(Device::GPU, type, {num_tokens, qkv_head_num, head_size});
	q_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
	k_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
	v_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});

	// transpose kv cache
	// why not kv_head_num?
	//	need repeat kv to adapt q head num
	// -> Broadcast: kv: [bs, q_head_num, max_k_len, head_size]
	//	如果MHA, 无需Broadcast
	k_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
	v_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});

	// q*k softmax qk * v
	// Attention: [bs, q_head_num, max_q_len, max_k_len] -> Qk*v gemm: [bs, q_head_num, max_q_len, head_size]
	qk_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
	qkv_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});

	// remove padding
	qkv_buf_wo_pad_1 = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});

	qkv_buf_wo_pad->data = allocator->Malloc(
		qkv_buf_wo_pad->data, sizeof(T) * num_tokens * qkv_head_num * head_size, false);
	q_buf_w_pad->data = allocator->Malloc(
		q_buf_w_pad->data, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size, false);
	k_buf_w_pad->data = (T*)q_buf_w_pad->data + head_num * batch_size * max_q_len * head_size;
	v_buf_w_pad->data = (T*)k_buf_w_pad->data + kv_head_num * batch_size * max_q_len * head_size;

	k_cache_buf->data = allocator->Malloc(
			k_cache_buf->data, 2 * sizeof(T) * batch_size * head_num * max_k_len * head_size, false);
	v_cache_buf->data = (T*)k_cache_buf->data + batch_size * head_num * max_k_len * head_size;

	// store qk and inplace store softmax output
	qk_buf->data =
			allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len, false);

	// store qk*v
	qkv_buf_w_pad->data = allocator->Malloc(
			qkv_buf_w_pad->data, sizeof(T) * batch_size * max_q_len * head_num * head_size, false);
	qkv_buf_wo_pad_1->data= allocator->Malloc(qkv_buf_wo_pad_1->data, sizeof(T) * num_tokens * head_num * head_size, false);
}