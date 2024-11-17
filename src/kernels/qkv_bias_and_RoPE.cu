// 注: Llama2中的RoPE, 是前一半与后一半为一组, 因此, 向量化全为错的

// 1. add qkv_bias to QKV, which has shape[num tokens, qkv head num, head_size], k head num = v head num
// 2. padding, QKV splits to q k v and their shape is [bs, q head num(kv head num), max q len, head_size]
// 3. rope and do attention
// 4. write back to global memory
// input:  QKV: [num_tokens, qkv_head_num, head_size]
//         qkv bias: [qkv_head_num, head_size]
// output: q: [bs, q_head_num, max_q_len, head_size]
//         k: [bs, kv_head_num, max_q_len, head_size]
//         v: [bs, kv_head_num, max q len, head_size]
// repeat kv
#include <math.h>
#include <stdio.h>
#include "src/kernels/qkv_bias_and_RoPE.h"

// hugging face
//    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
//         """Compute the inverse frequency."""
//         inv_freq = 1.0 / (base**(torch.arange(
//             0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
//                                  self.rotary_dim))
//         return inv_freq

//     def _compute_cos_sin_cache(self) -> torch.Tensor:
//         """Compute the cos and sin cache."""
//         inv_freq = self._compute_inv_freq(self.base)
//         t = torch.arange(self.max_position_embeddings,
//                          dtype=torch.float,
//                          device="cuda")

//         freqs = torch.einsum("i,j -> ij", t, inv_freq) // 外积
//         cos = freqs.cos() // 2048，64
//         sin = freqs.sin()
//         cache = torch.cat((cos, sin), dim=-1)
//         return cache

// cos与sin中的角度(seita)
inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step) {

    // 对应 HF 的 inv_freq
    float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

// 无向量化
inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef) {
    float2 rot_v; // rotary value
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;

    return rot_v;
}

// // RoPE公式决定必须做向量化
// // x0 x1 两两为一组共享seita(用f表示)
// // x0 * cos(mf0) - x1 * sin(mf0)
// // x1 * cos(mf0) + x0 * sin(mf0)
// inline __device__ float2 GetRoPEres(const float2 v, const float2 coef) {
//     float2 rot_v; // rotary value
//     rot_v.x = coef.x * v.x - coef.y * v.y;
//     rot_v.y = coef.x * v.y + coef.y * v.x;

//     return rot_v;
// }

// inline __device__ half2 GetRoPEres(const half2 v, const float2 coef) {
//     float2 fv = __half22float2(v);
//     float2 rot_fv = GetRoPEres(fv, coef);
    
//     return __float22half2_rn(rot_fv);
// }

// inline __device__ void apply_RoPE(half2 &q, half2 &k, 
//     int tid, int rot_embed_dim, float base, float t_step)
// {
//     if (2 * tid >= rot_embed_dim) {
//         return;
//     }

//     const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
//     q = GetRoPEres(q, coef);
//     k = GetRoPEres(k, coef);
// }

// // 两两一组, float4拆成 2 + 2
// inline __device__ void apply_RoPE(float4 &q, float4 &k, 
//     int tid, int rot_embed_dim, float base, float t_step)
// {
//     if (4 * tid >= rot_embed_dim) {
//         return;
//     }

//     // 两两一组, float4拆成 2 + 2
//     TwoFloat2 &q_ = *reinterpret_cast<TwoFloat2* >(&q);
//     TwoFloat2 &k_ = *reinterpret_cast<TwoFloat2* >(&k);

//     float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
//     q_.x = GetRoPEres(q_.x, coef0);
//     k_.x = GetRoPEres(k_.x, coef0);

//     float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
//     q_.y = GetRoPEres(q_.y, coef1);
//     k_.y = GetRoPEres(k_.x, coef1);
// }

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(
    T *q_buf, T *k_buf, T *v_buf, T *QKV,
    const T *qkv_bias,
    const int *padding_offset,      // created before qkv linear
    const int *history_length,
    const int *input_length,        // actual length of each seq
    const int batch_size,
    const int seq_len,              // max_seq_len to pad to
    const int token_num,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int rotary_embedding_dim,
    float rotary_embedding_base,    // default 10000 in llama
    int max_position_embeddings,    /*default 2048 in llama*/
    bool use_dynamic_ntk            /*placeholder for ntk RoPE*/)
{
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    
    // 为什么写 token_id 和 head_id -> 得到offset
    int token_id = blockIdx.x;
    int head_id  = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];

    int dst_token_id = token_id + token_padding_offset;
    int batch_id = dst_token_id / seq_len;          // 第几个seq_len
    int local_token_id = dst_token_id % seq_len;    // seq_len维度层面, 具体那一个token
    int qkv_head_num = head_num + 2 * kv_head_num;  // q + k + v(k = v) 为什么 q != k or v

    //         blockIdx.x   blockIdx.y  threadIdx.x
    // input: [num tokens, qkv head num, head size]
    // 计算input -> 因此不使用padding_offset后的id
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;

    // RoPE
    const int cur_seq_history_len = history_length[batch_id];
    const int time_step = cur_seq_history_len + local_token_id;
    if (tid >= rotary_embedding_dim / 2) {
        return; // 22一组的RoPE, 只需要前64个thread
    }
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, time_step);
    float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin); // 返回 x[0] 和 x[64]
    float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin); // 返回 x[0] 和 x[64]

    // wrtie back
    // output: q: [bs, q head num, max q len, head size]
    //         k: [bs, kv head num, max q len, head size]
    //         v: [bs, kv head num, max q len, head size]
    int dst_q_id  = batch_id * head_num * seq_len * head_size +
                    head_id * seq_len * head_size + 
                    local_token_id * head_size + tid;
    int dst_kv_id = batch_id * kv_head_num * seq_len * head_size +
                    head_id * seq_len * head_size + 
                    local_token_id * head_size + tid;

    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if (head_id < kv_head_num) {
        // for MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}

// template <>
// __global__ void add_fusedQKV_bias_transpose_kernel(
//     half *q_buf, half *k_buf, half *v_buf, half *QKV,
//     const half *qkv_bias,
//     const int *padding_offset,      // created before qkv linear
//     const int *history_length,
//     const int *input_length,        // actual length of each seq
//     const int batch_size,
//     const int seq_len,              // max_seq_len to pad to
//     const int token_num,
//     const int head_num,
//     const int kv_head_num,
//     const int head_size,
//     const int rotary_embedding_dim,
//     float rotary_embedding_base,    // default 10000 in llama
//     int max_position_embeddings,    /*default 2048 in llama*/
//     bool use_dynamic_ntk            /*placeholder for ntk RoPE*/)
// {
    // int vec_size = Vec<half>::size;
    // using Vec_t = typename Vec<half>::Type;
    // int token_id = blockIdx.x;
    // int head_id = blockIdx.y;
    // int tid = threadIdx.x;
    // int token_padding_offset = padding_offset[token_id];
    
    // // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
    // bool is_data = tid * vec_size < head_size;
    
    // // 1. prapare rebuilding , do rebuild padding and transpose when store
    // int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    // int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
    // int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id

    // // 2. bias add
    // int qkv_head_num = head_num + 2 * kv_head_num;
    // int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    // int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
    // int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
    // // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
    // // TODO: reduce the pointer converter and fuse for loop
    // Vec_t q, k, v;
    // if (is_data)
    // {
    //     q = *reinterpret_cast<Vec_t *>(&QKV[q_id]);
    //     Vec_t q_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size]));
    //     q = __hadd2(q, q_bias);
    // }
    // // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
    // if (is_data && head_id < kv_head_num)
    // {
    //     k = *reinterpret_cast<Vec_t *>(&QKV[k_id]);
    //     // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
    //     Vec_t k_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
    //     k = __hadd2(k, k_bias);
    //     v = *reinterpret_cast<Vec_t *>(&QKV[v_id]);
    //     Vec_t v_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
    //     v = __hadd2(v, v_bias);
    // }

    // // 3. RoPE
    // const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
    // const int context_length = cur_seq_history_len + input_length[batch_id];
    // const int timestep = cur_seq_history_len + local_token_id; //+ local_token_id得到m，即要结合history length做全局位置编码
    // // timestep为cos(m*theta)中的m

    // apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    // // 4.write back to gmem and do transpose
    // //  [bs, head num, seqlen, head size]
    // //  pay attention to local token id and kv head num and max_seq_len(seq_len)
    // int dst_q_id = batch_id * seq_len * head_num * head_size +
    //                head_id * seq_len * head_size +
    //                local_token_id * head_size + tid * vec_size;

    // int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
    //                 head_id * seq_len * head_size +
    //                 local_token_id * head_size + tid * vec_size;
    // if (is_data)
    // {
    //     *reinterpret_cast<Vec_t *>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
    //     if (head_id < kv_head_num)
    //     { // for MQA and GQA
    //         *reinterpret_cast<Vec_t *>(&k_buf[dst_kv_id]) = k;
    //         *reinterpret_cast<Vec_t *>(&v_buf[dst_kv_id]) = v;
    //     }
    // }
// }

template <typename T>
__global__ void rope_kernel_for_self_decoder(T *q, T *k, const int batch_szie, 
    const int head_num, const int kv_head_num, const int head_size, const int step, 
    int rotary_embedding_dim, float rotary_embedding_base) {
    
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_sride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_sride + tid;
    int k_offset = kv_batch_id * batch_stride + kv_head_id * head_sride + tid;

    if (tid >= rotary_embedding_dim / 2) {
        return;
    }
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1);
    float2 q_rotate = GetRoPEres(q[q_offset], q[q_offset + head_size / 2], cos_sin); // 返回 x[0] 和 x[64]
    float2 k_rotate = GetRoPEres(k[k_offset], k[k_offset + head_size / 2], cos_sin); // 返回 x[0] 和 x[64]

    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[q_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
    }

// TODO
template <>
__global__ void rope_kernel_for_self_decoder(half *q, half*k, 
    const int batch_szie, const int head_num, const int kv_head_num, const int head_size, const int step, 
    int rotary_embedding_dim, float rotary_embedding_base) {

    }

// input: [num tokens, qkv head_num, head size]
// output: q: [bs, head num, max q len, head size]
//       k/v: [bs, kv head num, max q len, head size]
template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(
    TensorWrapper<T> *q_buf,
    TensorWrapper<T> *k_buf,
    TensorWrapper<T> *v_buf,
    TensorWrapper<T> *QKV,
    BaseWeight<T> &qkv,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<int> *history_length,
    TensorWrapper<int> *input_length,
    LLaMAAttentionStaticParams &params)
{
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;

    // block只需要考虑head size, 来进行维度的分配
    dim3 grid(token_num, head_num);
    dim3 block(head_size);

    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(
            q_buf->data, k_buf->data, v_buf->data, QKV->data,
            qkv.bias, padding_offset->data, history_length->data, input_length->data,
            batch_size, seq_len, token_num, head_num, kv_head_num, head_size,
            params.rotary_embedding_dim, params.rotary_embedding_base, 
            params.max_position_embeddings, params.use_dynamic_ntk);
}

template void launchAddFusedQKVBiasTransposeAndRoPE(
    TensorWrapper<float> *q_buf, TensorWrapper<float> *k_buf, TensorWrapper<float> *v_buf,
    TensorWrapper<float> *QKV, BaseWeight<float> &qkv,
    TensorWrapper<int> *padding_offset, TensorWrapper<int> *history_length, TensorWrapper<int> *input_length,
    LLaMAAttentionStaticParams &params);
// template void launchAddFusedQKVBiasTransposeAndRoPE(
//             TensorWrapper<half> *q_buf, TensorWrapper<half> *k_buf, TensorWrapper<half> *v_buf,
//             TensorWrapper<half> *QKV, BaseWeight<half> &qkv,
//             TensorWrapper<int> *padding_offset, TensorWrapper<int> *history_length, TensorWrapper<int> *input_length,
//             LLaMAAttentionStaticParams &params);

template<typename T>
void launchRoPE(
    TensorWrapper<T> *qkv_buf, 
    TensorWrapper<int> *step, // 历史句子长度, 相当于const int time_step = cur_seq_history_len + local_token_id;
    LLaMAAttentionStaticParams &static_params)
{
    int head_num = 32; // only for Llama2
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();

    T *qkv_data = qkv_buf->data;
    T *q = qkv_data;
    T *k = qkv_data + head_num * head_size;

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int max_position_embeddnigs = static_params.max_position_embeddings;
    
    dim3 grid(head_num, batch_size);
    dim3 block(head_size);
    rope_kernel_for_self_decoder<T><<<grid, block>>>(
        q, k, batch_size, head_num, head_size, cur_step, 
        rotary_embedding_dim, rotary_embedding_base);
}