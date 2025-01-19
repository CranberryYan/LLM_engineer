#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator 
{
public:
    virtual ~BaseAllocator() {};
public:
    template<typename T>
    T *Malloc(T *ptr, size_t size, bool is_host) {
        return (T*)UnifyMalloc((void*)ptr, size, is_host);
    }

    template<typename T>
    void Free(T *ptr, bool is_host=false) {
        return UnifyFree((void*)ptr, is_host);
    }

    virtual void* UnifyMalloc(void *ptr, size_t size, bool is_host = false) = 0;

    virtual void UnifyFree(void *ptr, bool is_host = false) = 0;
};