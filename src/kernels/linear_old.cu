#include <iostream>
#include "src/kernels/linear_old.h"


// 1.
// A: weight: [hidden_units, hidden_units]
// B: input:  [seqlen, hidden_units]
// A * B = C   transb = false
// 2.
// A: input:  [bs, hidden_units]
// B: weight: [vocab_size, hidden_units]
// A * B^T = C   transb = true
// A: [m, k]   B: [k, n]   C: [m, n]
template <typename T>
void launchLinearGemm(TensorWrapper<T> *input, BaseWeight<T> &weight,
        TensorWrapper<T> *output, cublasWrapper* cublas_wrapper,
        bool trans_a, bool trans_b)
{
    int input_lda  = input->shape[0];

    // 虽然weight可能需要被转置, 但是还是传入第一个维度
    int weight_ldb = weight.shape[0];
    int weight_1st_dim = weight.shape[0];
    int weight_2nd_dim = weight.shape[1];

    int output_ldc = input_lda;
    int n = output->shape[1];

    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // A: input:  [bs, hidden_units]
    // B: weight: [vocab_size, hidden_units]
    // A * B^T = C   transb = true
    if (trans_b) {
        // [bs, 1, hidden_units] * [hidden_units, hidden_units]
        if (input->shape.size() > 2 ) {
            LLM_CHECK_WITH_INFO(input->shape[2] == weight.shape[1], "when trans_b, 3rd dim of input = 2nd dim of weight"); 
        }      
        LLM_CHECK_WITH_INFO(input->shape[1] == weight.shape[1], "when trans_b, 2nd dim of input = 2nd dim of weight");
    }

    // A: input   B: weight
    // A: [m, k]   B: [k, n]/[n, k](转置)   C: [m, n]
    // m: input_lda   n: output->shape[1]/weight_1st_dim(转置)   k: weight_ldb/weight_2nd_dim(转置)
    cublas_wrapper->Gemm(transA, transB, input_lda, trans_b ? weight_1st_dim : n, trans_b ? weight_2nd_dim : weight_ldb,
                        input->data, input_lda, weight.data, weight_ldb, output->data, output_ldc, 1.0f, 0.0f);
}

// 1. Q * K
// Q.shape: [bs, head_num, seq_len, hidden_units]
// K.shape: [bs, head_num, seq_len, hidden_units]
// Q * K^T = Score
// 2. Score * V
// Score.shape: [bs, head_num, seq_len, seq_len]
// V.shape: [bs, head_num, seq_len, hidden_units]
// Score * V = output.shape: [bs, head_num, seq_len, hidden_units]
template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1, TensorWrapper<T> *input2, TensorWrapper<T> *output, 
        cublasWrapper* cublas_wrapper, bool trans_a, bool trans_b)
{
    // input: [bs, head_nums, seqlen, head_size]
    // A: input   B: weight
    // A: [m, k]   B: [k, n]/[n, k](转置)   C: [m, n]
    int Am = input1->shape[2];
    int Ak = input1->shape[3];
    int Bk = input2->shape[2];
    int Bn = input2->shape[3];

    int lda = Am;
    int ldb = Bk;
    int ldc = Am;

    int64_t strideA = Am * Ak;
    int64_t strideB = Bk * Bn;
    int64_t strideC = Am * Bn;

    // 有多少个后两维的矩阵在做乘法
    int batchCount = input1->shape[0] * input1->shape[1];
    
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->strideBatchGemm(transA, transB, Am, trans_b ? Bk : Bn, Ak, 
                        input1->data, lda, strideA, 
                        input2->data, ldb, strideB,
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