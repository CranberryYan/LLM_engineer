# pragma once
# include <string>
# include <fstream>
# include <iostream>
# include <cuda.h>
# include <cuda_runtime.h>
# include <cublas_v2.h>


//(RussWong) note: some macro check to assert for helping us find errors, so that we can 
# define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void llmAssert(bool result, const char* const file, int const line, std::string const& info = "")
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

// 检查 tensor 是在 host 还是 device
# define LLM_CHECK(val) llmAssert(val, __FILE__, __LINE__)
# define LLM_CHECK_WITH_INFO(val, info)                                                                                 \
    do {                                                                                                                \
        bool is_valid_val = (val);                                                                                      \
        if (!is_valid_val) {                                                                                            \
            llmAssert(is_valid_val, __FILE__, __LINE__, (info));                                                        \
        }                                                                                                               \
    } while (0)

#define DeviceSyncAndCheckCudaError() syncAndCheck(__FILE__, __LINE__)

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}