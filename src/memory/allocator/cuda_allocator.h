#pragma once
#include <map>
#include <vector>
#include <iostream>
#include <string.h>
#include "base_allocator.h"

struct CudaBigBlock {
public:
    CudaBigBlock() = default;
    CudaBigBlock(void *data, size_t size, bool is_allocated) :
        data(data), size(size), is_allocated(is_allocated){ }
public:
    void *data;
    size_t size;
    bool is_allocated;
};

struct CudaSmallBlock {
public:
    CudaSmallBlock() = default;
    CudaSmallBlock(void *data, size_t size, bool is_allocated) :
        data(data), size(size), is_allocated(is_allocated){}
public:
    void *data;
    size_t size;
    bool is_allocated;
};

class CudaAllocator : public BaseAllocator
{
public:
    CudaAllocator() {
        cudaGetDevice(&device_id);
    }

    ~CudaAllocator() {
        // it: 遍历所有的device
        //  无论是cudaSmallBlocksMap or cudaBigBlocksMap
        //  iter数量是一样的(device数量), 因为key是device_id
        for (auto &it: cudaSmallBlocksMap) {
            // 当前device中的所有smallblock, 是一个vector
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                cudaFree(cudaBlocks[i].data);
            }

            // it.first: device_id
            // bigBlocks: 当前device中所有的bigblock, 是一个vector
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                cudaFree(bigBlocks[i].data);
            }            
        }
    }
public:
    void* UnifyMalloc(void *ptr, size_t size, bool is_host) {
        size = ((size + 31) / 32) * 32; // 对齐, FP32
        
        // 1. host
        if (is_host) {
            ptr = malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }

        // 2. big buf
        //  先去bigblocks里面找空闲的(free出来且未归还到OS)
        if (size > 1024 * 1024) {
            int block_id = -1;
            auto &BigBlocks = cudaBigBlocksMap[device_id];
            for (int i = 0; i < BigBlocks.size(); ++i) {
                if (BigBlocks[i].size > size && 
                    !BigBlocks[i].is_allocated && 
                    BigBlocks[i].size - size < 1024 * 1024) {
                    if (block_id == -1 || 
                        BigBlocks[block_id].size > BigBlocks[i].size) {
                        block_id = i;
                    }
                }
            }
            if (block_id != -1) { // 找到合适的block
                BigBlocks[block_id].is_allocated = true;
                return BigBlocks[block_id].data;
            } else { // 没找到合适的block, 自行malloc
                void *newBuffer = (void*) ptr;
                CHECK(cudaMalloc((void**)&newBuffer, size));
                CHECK(cudaMemset(newBuffer, 0, size));
                BigBlocks.push_back(CudaBigBlock(newBuffer, size, false));
                return newBuffer;
            }
        }

        // 3. small buf
        //  先去smallblocks里面找空闲的(分配出来但是没有具体任务的内存地址)
        else {
            auto &SmallBlocks = cudaSmallBlocksMap[device_id];
            for (int i = 0; i < SmallBlocks.size(); ++i) {
                if (SmallBlocks[i].size > size && 
                    !SmallBlocks[i].is_allocated) { // 找到合适的block
                    SmallBlocks[i].is_allocated = true;
                    FreeSize[i] += SmallBlocks[i].size; // ???
                    // FreeSize[device_id] += SmallBlocks[i].size; // 不同device, 总体计算
                    return SmallBlocks[i].data;
                }
            }

            // 没找到合适的block, 自行malloc
            void* newBuffer = (void*) ptr;
            CHECK(cudaMalloc((void**)&newBuffer, size));
            CHECK(cudaMemset(newBuffer, 0, size));
            SmallBlocks.push_back(CudaSmallBlock(newBuffer, size, false));
            return newBuffer;
        }
    }

    void UnifyFree(void *ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }

        // 1. is_host
        if (is_host) {
            free(ptr);
            return;
        }

        // 2. 清理碎片: 累计的smallbuf超出1G, 清理未分配出去的smallblocks并归还OS
        //      已经被分配的继续保留, 重新排列, 更紧凑
        // 遍历所有device
        for (auto &it : cudaSmallBlocksMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> tmp;
                for (int i = 0; i < cudaBlocks.size(); ++i) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        cudaFree(cudaBlocks[i].data);
                        // cudaBlocks[i].is_allocated = false; // ??? 为什么不置为false, 因为判断条件...
                    } else {
                        tmp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = tmp;
                FreeSize[it.first] = 0;
            }
        }

        // 3. 找到待free的buffer位置, is_allocated = false, 但并不归还, 
        //  等要用的时候在拿出来用
        // it: 遍历所有的device
        //  无论是cudaSmallBlocksMap or cudaBigBlocksMap
        //  iter数量是一样的(device数量), 因为key是device_id
        for (auto &it : cudaSmallBlocksMap) {
            // cudaBlocks: 当前device中的所有smallblock, 是一个vector
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); ++i) {
                if (cudaBlocks[i].data == ptr) {
                    cudaBlocks[i].is_allocated = false;
                    FreeSize[it.first] += cudaBlocks[i].size; 
                    // cudaFree(cudaBlocks[i].data); // 并不free
                    return;
                }
            }

            // bigBlocks: 当前device中的所有bigblock, 是一个ector
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }
        // for (auto &it : cudaBigBlocksMap) {
        //     auto &cudaBlocks = it.second;
        //     for (int i = 0; i < cudaBlocks.size(); ++i) {
        //         if (cudaBlocks[i].data == ptr) {
        //             cudaBlocks[i].is_allocated = false;
        //             // cudaFree(cudaBlocks[i].data); // 并不free
        //             return;
        //         }
        //     }
        // }

        // 4. cudaFree
        cudaFree(ptr);
    }
private:
    int device_id;
    std::map<int, size_t> FreeSize;
    std::map<int, std::vector<CudaBigBlock>> cudaBigBlocksMap;
    std::map<int, std::vector<CudaSmallBlock>> cudaSmallBlocksMap;
    size_t total_allocated_size = 0;  
    int dev_id;
};