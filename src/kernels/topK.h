#pragma once
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/utils/tensor.h"

template<typename T, int K>
struct topK
{
public: // 默认就是public
    T val[K];
    int id[K];

    __device__ void init() {
        for (int i = 0; i < K; ++i) {
            id[i] = -1;
            val[i] = FLT_MIN;
        }
    }

    // 构建大顶推, 最大的在开头
    __device__ void insertHeap(T data, int data_id) {
        // id[K - 1] == -1
        //  插进去的是第一个元素, 因为此时最后一个id都为-1(还是初始化的值)
        // val[K - 1] < data
        //  插进去的元素不是第一个元素, 大于当前最小的元素
        if (id[K - 1] == -1 || val[K - 1] < data) {
            id[K - 1] = data_id;
            val[K - 1] = data;
        }

        // K - 2: 正常的冒泡排序需要K - 1次, 但是在新数据插入之前, 已经是大顶推
        // 所以从末尾开始比, 冒泡排序, 看这个新数据能走到哪个位置
        for (int i = K - 2; i >= 0; --i) {
            if (val[i + 1] > val[i]) {
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};

template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
    TensorWrapper<int> *topk_ids, TensorWrapper<T> *topk_vals,
    TensorWrapper<int> *final_topk_ids, TensorWrapper<T> *final_topk_vals);