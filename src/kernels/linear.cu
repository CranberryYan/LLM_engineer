// linear: 用于gemm, 2d x 2d: fused
//          batch_gemm, 4d x 4d: [batch_size, head_num, seq_len, head_size]: Qk_gemm, Qk*v_gemm
//          LMHead(linear): hidden_units -> vocab_size(接下来要sampling)
// input: [num_tokens] -> input_embedding: [num_tokens, hidden_size]
//                              |
//                              -> cal_paddingoffset: [bs, max_num_tokens, hidden_size]
//                              |
//                              -> build_casual_mask: mask: [bs, max_num_tokens, max_num_tokens]
//                              |
//                              -> RMSNorm: [num_tokens, hidden_size] -> fusedQkvGemm: [num_tokens, hidden_size] * [hidden_size, hidden_size] -> [num_tokens, hidden_size]
#include <iostream>
#include "src/kernels/linear_old.h"


// 1.
// A: weight: [hidden_units, hidden_units]
// B: input:  [seqlen, hidden_units]
// A * B = C   transb = false
// 2.
// A: input:  [num_tokens, hidden_units]
// B: weight: [vocab_size, hidden_units]
// A * B^T = C   transb = true
// A: [m, k]   B: [k, n]   C: [m, n]

// ctx qkv linear: [num_tokens, hidden_units] * [hidden_units, hidden_untis]
// ctx attn output linear: [num_tokens, head_num, head_size] * [hidden_units, hidden_units] // hidden_units = head_num * head_size
// self qkv linear: [bs, hidden_units] * [hidden_units, hidden_units] = [bs, head_num, head_size, hidden_units]
// self attn output linear: [bs, hidden_units] * [hidden_units, hidden_units]
// lmhead linear: [bs, hidden_units] * [vocab_size, hidden_units](转置)
// gate: [bs/token_nums, hidden_units] * [hidden_units, inter_size]
// up:   [bs/token_nums, hidden_units] * [hidden_units, inter_size]
// fusedGateUpGemm: [bs/token_nums, hidden_units] * [hidden_units, 2 * inter_size] = [bs/token_nums, 2, inter_size]
// 2 * inter_size: gate_linear 和 up_linear 的 weight进行水平拼接
// down: [bs/token_nums, inter_size] * [hidden_units, inter_size]
template <typename T>
void launchLinearGemm(TensorWrapper<T>* input, BaseWeight<T> &weight,
    TensorWrapper<T>* output, cublasWrapper* cublas_wrapper,
    bool trans_a, bool trans_b) {
    // row major: y   = x   * w
    // col major: y^T = w^T * x^T
    // trans_b: false: y^T = w^T * x^T
    // trans_b: true:  y^T = w   * x^T
    // trans_b实际控制的是A
    int Am = weight.shape[1];
    int Ak = weight.shape[0];
    int Bk = input->shape[1];
    int Bn = input->shape[0];
    int Cm = output->shape[1];
    int Cn = output->shape[0];

    // 输入为3维
    Bk = input->shape.size() == 3 ? input-> shape[1] * input->shape[2] : input->shape[1];
    
    // 输出为3维
    Cm = output->shape.size() == 3 ? output->shape[1] * output->shape[2] : output->shape[1];

    int lda = Am;
    int ldb = Bk;
    int ldc = Am;

    // trans_b: 将A转置(weight)
    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (!trans_b && !trans_a) {
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input == 1st dim of weight");
    }

    // A: input   B: weight
    // A: [m, k]   B: [k, n]/[n, k](转置)   C: [m, n]
    // m: input_lda   n: output->shape[1]/weight_1st_dim(转置)   k: weight_ldb/weight_2nd_dim(转置)
    cublas_wrapper->Gemm(
        transA, transB, 
        trans_b ? Ak : Am, Cn, Bk, 
        weight.data, lda, input->data, 
        ldb, output->data, ldc, 1.0f, 0.0f);
}

// 1. Q * K
// Q.shape: [bs, head_num, length_q, head_size]
// K.shape: [bs, head_num, length_k, head_size]
// Q * K^T = Score
// 2. Score * V
// Score.shape: [bs, head_num, length_q, length_k]
// V.shape: [bs, head_num, length_k, head_size]
// Score * V = output.shape: [bs, head_num, length_k, head_size]
template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T>* input1, TensorWrapper<T>* input2, TensorWrapper<T>* output, 
        cublasWrapper* cublas_wrapper, bool trans_a, bool trans_b) {
    int An = input2->shape[3]; // head_size // head_size
    int Ak = input2->shape[2]; // length_k  // length_k
    int Bk = input1->shape[3]; // head_size // length_k
    int Bm = input1->shape[2]; // length_q  // length_q
    int Cm = output->shape[2]; // length_q  // length_q
    int Cn = output->shape[3]; // length_k  // head_size

    int lda = An;
    int ldb = Bk;
    int ldc = Cn;

    int64_t strideA = An * Ak;
    int64_t strideB = Bk * Bm;
    int64_t strideC = Cn * Cm;

    // 有多少个后两维的矩阵在做乘法
    int batchCount = input1->shape[0] * input1->shape[1];
    
    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->strideBatchGemm(
        transA, transB, Cn, Cm, Bk, 
        input2->data, lda, strideA, 
        input1->data, ldb, strideB,
        output->data, ldc, strideC, 
        batchCount, 1.0f, 0.0f);
}

template void launchLinearGemm(TensorWrapper<float> *input, BaseWeight<float> &weight, TensorWrapper<float> *output,
                               cublasWrapper *cublas_wrapper, bool trans_a, bool trans_b);

template void launchLinearGemm(TensorWrapper<half> *input, BaseWeight<half> &weight, TensorWrapper<half> *output,
                               cublasWrapper *cublas_wrapper, bool trans_a, bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<float> *input1, TensorWrapper<float> *input2, TensorWrapper<float> *output,
                                           cublasWrapper *cublas_wrapper, bool trans_a, bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<half> *input1, TensorWrapper<half> *input2, TensorWrapper<half> *output,
                                           cublasWrapper *cublas_wrapper, bool trans_a, bool trans_b);