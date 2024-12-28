/*
Fused decoder attention draft

Biasadd + rope + concat kv + repeat_kv + qk gemm + softmax + qk * v gemm
Key points:
    1. How to fuse: RoPE后的q k 都在显存上, 很耗时 -> 关键在于复用reg和smem中的数据
    2. How to allocate and use dynamic shared mem
    3. q * k batch Gemv: [bs, head_num, 1, step]
        q.shape: [bs, head_num, 1, head_size]   k.shape: [bs, head_num, step, head_size]
        如果transpose会引入开销, 所以不transpose, 而是 q 的一行乘 k 的一行
        然后累加head_size个[1, step]行向量
    4. qk * v Gemv: [bs, head_num, 1, head_size]
        qk.shape: [bs, head_num, 1, step]
        v.shape: [bs, head_num, step, head_size]
        output.shape: [bs, head_num, 1, had_size] 和 q.shape 同
        如果使用matmul(行 * 列)性能会很差, 
        因为按照列的方式访问v矩阵, 如果head_size(矩阵的列数)比较大, 访问第一行第一列的元素后访问第二行第一列元素会 cache miss
        cache miss: v一开始在显存, 然后加载到L2 cache
        即使head_size不大, 不会在访问第二行就cache miss, 但是在后面的几行终究会miss
        即使step也不大, cache中也会有很多暂时用不到的数据(第一行第2个到最后一个元素)
        总上, 无论如何都不好
        解决办法:
            (1)img2col, 内存重排, 但是需要transpose, 会引入额外开销, 但是要小于上述开销(cache miss 造成的延时)
            (2)先不着急求最终结果, 先求中间结果, 不断累加, 避免跨head_size访问其他元素
        eg:
            step: 5   head_size: 3
            qk: [1, 2, 3, 4, 5]
            v: [[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5]]
            qk的第一个元素 * v 的第一行 -> 1 * [1, 1, 1]
            qk的第二个元素 * v 的第二行 -> 2 * [4, 4, 4]
                            .
                            .
                            .
            qk的第五个元素 * v 的第五行 -> 5 * [25, 25, 25]
            将其累加 -> [[55, 55, 55]
*/
#include <math.h>
#include <stdio.h>
#include "fused_decoder_self_attention.h"


template<typename T>
__device__ T warpReduceSum(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;

}

template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];//why add static?or will report incomplete type
    // returned val is the sum computed by 0th thread.
    val = warpReduceSum<T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);

}

template<typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    // returned val is the max computed by 0th thread.
    val = warpReduceMax(val); // remove <T> can ignore the multi-overloaded error?
    //note: here return val of warpreducemax should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

// kv cache is the output of context attention(prompt phase)
// and the intput of masked attention(token phase)
// struct masked_MHA_kernel_params {
// float *q;        // [bs, q_num_head, 1, head_size] 1: 自回归
// flaot *k;        // [bs, kv_num_head, step/max_seq_len, head_size]
// flaot *v;        // [bs, kv_num_head, step/max_seq_len, head_size]
// float *k_cache;  // output, [num_layers, bs, kv_num_head, step/max_seq_len, head_size] from prompt phase
// float *v_cache;  // output, [num_layers, bs, kv_num_head, step/max_seq_len, head_size] from prompt phase
// int batch_size;
// int num_heads;
// int head_size;
// flaot scales;
// int step;
// flaot *mha_output;
// }
template<typename T>
__global__ void masked_MHA_kernel(T* q, T *k, T *v,
    T *qkv_bias, T *k_cache, T *v_cache, T *mha_output,
    const int batch_size, const int head_num,
    const int kv_head_num, const int max_seq_len,
    const int head_size, const int step,
    int rotary_embedding_dim, float rotary_embedding_base) {

    // head_num / kv_head_num: 一个q_head对应多个kv_head
    // eg: q_head_num: 8   kv_head_num: 2
    // q_head_id: 0~3: kv_head_id: 0   q_head_id: 4~7: kv_head_id: 1head_num / kv_head_num: 一个q_head对应多个kv_head
    // eg: q_head_num: 8   kv_head_num: 2
    // q_head_id: 0~3: kv_head_id: 0   q_head_id: 4~7: kv_head_id: 1
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;

	int step_stride = head_size;
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;

    // q: [bs, q_num_head, 1, head_size]
    // k: [bs, kv_num_head, step/max_seq_len, head_size]
    int q_offset_vec = q_batch_id * batch_stride + 
			q_head_id * head_stride + 
			tid * vec_size; // 以vec_stride为stride进行偏移
    int k_offset_vec = kv_batch_id * kv_batch_stride + 
			kv_head_id * head_stride + 
			tid * vec_size;
    
    // k_cache: [num_layers, bs, kv_num_head, step/max_seq_len, head_size]
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size + 
			kv_head_id * max_seq_len * head_size + 
			tid * vec_size; // max_seq_len始终为1, seq_len_id始终为0, step由gemm时指定

    // biasadd & RoPE
    // 重命名, 表示qkv都在显存中
    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;
    Vec_t qvec, kvec, vvec;
    
    // qvec是向量, q_mem是标量 -> 类型转换
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec]));
        Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]); // qkv_bias: [qkv_head_num, head_size]
		qvec.x += q_bias.x;
		qvec.y += q_bias.y;
		qvec.z += q_bias.z;
		qvec.w += q_bias.w;

        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec]));
        Vec_t k_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[head_num * head_size + kv_head_id * head_size + tid * vec_size]);
		kvec.x += q_bias.x;
		kvec.y += q_bias.y;
		kvec.z += q_bias.z;
		kvec.w += q_bias.w;

        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec]));
        Vec_t v_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[head_num * head_size + kv_head_id * head_size + tid * vec_size]);
		vvec.x += q_bias.x;
		vvec.y += q_bias.y;
		vvec.z += q_bias.z;
		vvec.w += q_bias.w;
    }

    // 处理shared_mem
    extern __shared__ char sqk[]; // 根据smem_size_bytes动态分配

    // dyn smem plan
    // q: [bs, q_num_head, 1, head_size]
    // 只需要保存[1, head_size], head_size个元素
    // k: [bs, kv_num_head, step/seqlen, head_size]
    // 只需要保存[step/seqlen, head_size] -> head_size个元素
    // 因为kv是每次取一行来与q的一个元素做运算, 因此每次也是head_size个元素
    // 综上, 其实kv没必要非要用smem, 因为kv每次读一行, 一般都会进入cache
    // q也没必要用smem, 因为当前block只处理一行kv, q每次只和一行kv进行运算, 如果q在一个block中一次和多行kv进行运算, 才有复用的价值
    // size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);
	float scale = rsqrt((float)head_size);
	T *sq_scalar = reinterpret_cast<T*>(sqk);
    float *logits = reinterpret_cast<float*>(sq_scalar + head_size);
	Vec_t *sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads(); // 对smem做操作, 需要同步

	// add
	float zero = 0.0f;
	Vec_t zero_f4 = scalar_cast_vec<Vec_t>(zero);
	Vec_t scale_f4 = scalar_cast_vec<Vec_t>(scale);

    // q * k gemm
    // 一个block循环计算step行
    // TODO: 多个blokc并行计算step行
    for (int iter = 0; iter < step; ++iter) {
		// cache_offset: 省略了行索引   iter * step_stride: 遍历每一行
        Vec_t kvec_qk = tid * vec_size < head_size
				? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset])
				: zero_f4; // 此时全部为向量化的操作, 不可使用标量0

        if (iter == step - 1 && tid * vec_size < head_size) {
            // kvec: 在显存中的k, 被读取成向量
            // iter: step_id
            // int cache_offset = batch_size * kv_batch_stride -> batch_size * kv_head_num * head_size;
            // k_cache: [bs, kv_num_head, step/max_seq_len, head_size]
            
			// 在此之前的kvec_qk其实没读到最后一次iter中的k_cache, 因为现在才被赋值
			// 所以需要我们自行给值
			*reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }
        
        // mul
		Vec_t qk = zero_f4;
		qk.x = tid * vec_size < head_size 
				? sq[tid].x * kvec_qk.x * scale_f4.x
				: 0;
		qk.y = tid * vec_size < head_size 
				? sq[tid].y * kvec_qk.y * scale_f4.y
				: 0;
		qk.z = tid * vec_size < head_size 
				? sq[tid].z * kvec_qk.z * scale_f4.z
				: 0;
		qk.w = tid * vec_size < head_size 
				? sq[tid].w * kvec_qk.w * scale_f4.w
				: 0;

		T qk_acc = qk.x + qk.y + qk.w + qk.z;

        // reduce sum
        T atten_score = blockReduceSum<T>(qk_acc);
        if (tid == 0) {
            logits[iter] = atten_score;
        }
        __syncthreads(); // 对smem做操作, 需要同步
    }

    // softmax
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;

    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads(); // 对smem做操作, 需要同步

    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0) {
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads(); // 对smem做操作, 需要同步

    logits[tid] = tid < step ? (T)(fenzi / fenmu) : 0;
    __syncthreads(); // 对smem做操作, 需要同步

    // logits * v
    if (tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t>(0.0f);
        for (int iter = 0; iter < step; ++iter) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);

            if (iter == step - 1) {
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
			}

            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
		}

		*reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
	}
}

template<>
__global__ void masked_MHA_kernel(half* q, half* k, half* v,
	half* qkv_bias, half* k_cache, half* v_cache, half* mha_output,
	const int batch_size, const int head_num, const int kv_head_num,
	const int max_seq_len, const int head_size, const int step,
	int rotary_embedding_dim, float rotary_embedding_base) { 







}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T> *qkv_buf, BaseWeight<T> &qkv,
    TensorWrapper<int> *layer_id, TensorWrapper<T> *k_cache, TensorWrapper<T> *v_cache,
    TensorWrapper<bool> *finished,  // 生成token是否结束
    TensorWrapper<int> *step,       // 生成到第几个token
    TensorWrapper<T> *mha_output,   // 输出
    LLaMaAttentionStaticParams &static_params/*关于RoPE的参数*/) {

    // 输入: qkv_buf: 来自于qkv_linear的输出
    // qkv:  qkv_linear的weight中的bias
    // qkv_buf: [bs, qkv_head_num, head_size]
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int head_size = qkv_buf->shape[2];

    // k_cache: num_layers, bs, max_seq_len, kv_num_head, head_size]
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3]; 
    int head_num = qkv_head_num - 2 * kv_head_num; // q_head_num


    // inline T getVal() const {
    //     LLM_CHECK(location == CPU);
    //     return getVal(0);
    // }
    // 因此, step和layer_id在cpu上面, 因为这些都是标量
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);

    //qkv_buf: [bs, 1, qkv_head_num, head_size]'
    T *qkv_data = qkv_buf->data;
    T *q = qkv_data;
    T *k = qkv_data + head_num * head_size;
    T *v = qkv_data + (head_num + kv_head_num) * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    // int   max_position_embeddings = static_params.max_position_embeddings;
    // bool  use_dynamic_ntk = static_params.use_dynamic_ntk;
    
    dim3 grid(head_num * batch_size);
    dim3 block(head_size);

    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(
        q, k, v, qkv.bias, k_cache->data + layer_offset, v_cache->data + layer_offset,
        mha_output->data, batch_size, head_num, kv_head_num, 
        max_seq_len, head_size, cur_step,
        rotary_embedding_dim, rotary_embedding_base);
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf, BaseWeight<float>& qkv,
    TensorWrapper<int>* layer_id, TensorWrapper<float>* k_cache, TensorWrapper<float>* v_cache,
    TensorWrapper<bool>* finished, TensorWrapper<int>* step, TensorWrapper<float>* mha_output,
    LLaMaAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf, BaseWeight<half>& qkv,
    TensorWrapper<int>* layer_id, TensorWrapper<half>* k_cache, TensorWrapper<half>* v_cache,
    TensorWrapper<bool>* finished, TensorWrapper<int>* step, TensorWrapper<half>* mha_output,
    LLaMaAttentionStaticParams& static_params);