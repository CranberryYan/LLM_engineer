#include <iostream>
#include <cub/cub.cuh>
#include "src/kernels/topK.h"


template<typename T, int K>
__device__ topK<T, K> reduce_functor(const topK<T, K> &a, const topK<T, K> &b) {
    topK<T, K> res = a;
    for (int i = 0; i < K; ++i) {
        res.insertHeap(b.val[i], b.id[i]);
    }

    return res;
}

// 分两轮去排, vocab_size: 32k, 太大了
// 
// gridsize: bs * beam_width * BlockPerBeam
// blockSize: 256
// shape infer: [bs, beam_width, vocab_size] -> [bs, beam_width, BLockPerBeam, K]
//  vocab_size -> [BLockPerBeam, K]
//  eg:
//      |-------|-------|-------|-------|-------|
//      将vocab_size分割成BLockPerBeam段, 每一段由一个单独的block来处理 -> 每一段(block)都会有一组topK
template <typename T, int K, int blockSize, int BlockPerBem>
__global__  void topK_kernel_round1(const T *probs, const int vocab_size,
    int *topK_ids, T *topK_vals) {
    
    // 输出: [bs * beam_width *  BLockPerBeam, K]
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row_id = bid / BlockPerBem;     // 位于哪一段
    int block_lane = bid % BlockPerBem; // 每一段由一个block负责, 当前处于这一段的什么位置
    topK<T, K> thread_topK;
    thread_topK.init();

    for (int data_id = tid + block_lane * blockSize; 
        data_id < vocab_size; data_id += BlockPerBem * blockSize) {
        
        // gridSize: bs * beam_width * BlockPerBeam
        // input: [bs, beam_width, vocab_size]
        // row_id: bid / BlockPerBem -> bs * beam_width
        int data_offset = row_id * vocab_size + data_id;
        T data = probs[data_offset];
        thread_topK.insertHeap(data, data_id);
    }
    
    // 初始化
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    
    // 申请；临时空间
    __shared__ typename blockreduce::TempStorage tmp_storage;

    // 调用
    topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    if (tid == 0) {
        for (int k_offset = 0; k_offset < K; ++k_offset) {
            
            // output: [bs, beam_width, BLockPerBeam, K]
            // row_id: bid / BlockPerBem -> bs * beam_width
            int dst_offset = row_id * BlockPerBem * K + block_lane * K + k_offset;
            topK_ids[dst_offset] = block_topk.id[k_offset];
            topK_vals[dst_offset] = block_topk.val[k_offset];
        }
    }
}

// gridSize: bs
// blockSize: 256
// shape infer: [bs, beam_width, BlockPerBeam, K] -> [bs, beam_width, K]
template <typename T, int beam_width, int K, int blockSize, int BlockPerBeam>
__global__  void topK_kernel_round2(const int *topK_ids, const T *topK_vals,
    int *final_topK_ids, T *final_topK_vals) {
    
    // 输出: [bs * beam_width *  BLockPerBeam, K]
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row_id = bid;                   // 位于哪一段
    topK<T, K> thread_topK;
    thread_topK.init();

    for (int data_id = tid; data_id < beam_width * BlockPerBeam * K; data_id += blockSize) {
        
        // gridSize: bs
        // input: [bs, beam_width, BlockPerBeam, K]
        int data_offset = bid * beam_width * BlockPerBeam * K + data_id;
        thread_topK.insertHeap(topK_vals[data_offset], topK_ids[data_offset]);
    }
    
    // 初始化
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    
    // 申请；临时空间
    __shared__ typename blockreduce::TempStorage tmp_storage;

    // 调用
    topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    if (tid == 0) {
        for (int k_offset = 0; k_offset < K; ++k_offset) {

            // output: [bs, beam_width, K]
            int beam_id = (blockDim.x * blockIdx.x + tid) / BlockPerBeam / K;
            int dst_offset = bid * beam_width * K + k_offset;
            final_topK_ids[dst_offset] = block_topk.id[k_offset];
            final_topK_vals[dst_offset] = block_topk.val[k_offset];
        }
    }
}

template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             TensorWrapper<int> *topk_ids,
                             TensorWrapper<T> *topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals)
{
    // support both beamserach and sampling topk by integrate beamwidth into batchsize, we get variable bsxbw = bs*bw, 
    // the probs shape is [bs*bw, vocabsize]
    int batch_size = probs->shape[0];
    int vocab_size = probs->shape[1];
    constexpr int BlockPerBeam = 8;
    constexpr int beamwidth = 1;
    constexpr int K = 5;
    
    // // buffer size
    // int topK_val_buf_size = batch_size * BlockPerBeam * K;
    // int topK_ids_buf_size = batch_size * BlockPerBeam * K;
    // int final_topK_val_buf_size = batch_size * K;
    
    T *topK_vals = topk_vals->data;
    int *topK_ids = topk_ids->data;
    T *final_topK_vals = final_topk_vals->data;
    int *final_topK_ids = final_topk_ids->data;    

    int maxBlockNums = 1024;
    int BlockNums1 = std::min(batch_size * beamwidth * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(batch_size, maxBlockNums);
    dim3 grid_round1(BlockNums1);
    dim3 block_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 block_round2(256);
    topK_kernel_round1<T, K, 256, BlockPerBeam>
        <<<grid_round1, block_round1>>>(probs->data, vocab_size, topK_ids, topK_vals);
    topK_kernel_round2<T, beamwidth, K, 256, BlockPerBeam>
        <<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
}

template void launchTopKforBeamSearch(TensorWrapper<float> *probs,
    TensorWrapper<int> *topk_ids, TensorWrapper<float> *topk_vals,
    TensorWrapper<int> *final_topk_ids, TensorWrapper<float> *final_topk_vals);

template void launchTopKforBeamSearch(TensorWrapper<half> *probs,
    TensorWrapper<int> *topk_ids, TensorWrapper<half> *topk_vals,
    TensorWrapper<int> *final_topk_ids, TensorWrapper<half> *final_topk_vals);