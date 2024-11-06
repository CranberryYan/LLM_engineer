#include <iostream>
#include "cublas_utils.h"



cublasWrapper::cublasWrapper(cublasHandle_t cublas_handle,
                                 cublasLtHandle_t cublaslt_handle):
    cublas_handle(cublas_handle),
    cublaslt_handle(cublaslt_handle) { }

cublasWrapper::~cublasWrapper() {}

void cublasWrapper::setFP32GemmConfig() {
    Atype       = CUDA_R_32F;
    Btype       = CUDA_R_32F;
    Ctype       = CUDA_R_32F;
    Computetype = CUDA_R_32F;
}

void cublasWrapper::setFP16GemmConfig() {
    Atype       = CUDA_R_16F;
    Btype       = CUDA_R_16F;
    Ctype       = CUDA_R_16F;
    Computetype = CUDA_R_16F;
}

// A: [m, k]   B: [k, n]   C: [m, n]
// lda: leading dimension 第一维
void cublasWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb,
        const int m, const int n, const int k, 
        const void* A, const int lda, const void* B, const int ldb, 
        void* C, const int ldc, float f_alpha, float f_beta)
{
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    int is_fp16_computeType = Computetype == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(h_alpha)) : reinterpret_cast<void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&(h_beta)) : reinterpret_cast<void*>(&f_beta);
    cublasGemmEx(cublas_handle, transa, transb, m, n, k,
        alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
        Computetype, CUBLAS_GEMM_DEFAULT);
}

// strideA: 
//  A: [1, 2, 3, 4]
//  strideA: 3 * 4
//  batch_count: 1 * 2
void cublasWrapper::strideBatchGemm(cublasOperation_t transa, cublasOperation_t transb,
        const int m, const int n, const int k,
        const void* A, const int lda, const int64_t strideA, 
        const void* B, const int ldb, const int64_t strideB, 
        void* C, const int ldc, const int64_t strideC, 
        const int batchCount, float f_alpha, float f_beta)
{
    int is_fp16_computeType = Computetype == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(f_alpha)) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&(f_beta)) : reinterpret_cast<const void*>(&f_beta);
    cublasGemmStridedBatchedEx(cublas_handle, transa, transb, m, n, k,
        alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta,
        C, Ctype, ldc, strideC, batchCount, Computetype, CUBLAS_GEMM_DEFAULT);
}