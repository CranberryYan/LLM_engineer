// input_embedding: 输入转换为词向量

// input: [num_tokens, hidden_size]
# include <iostream>
# include "input_embedding.h"


// input: [token_nums]
// output: [token_nums, hidden_size]
template<typename T>
__global__ void embeddingFunctor(const int* input_ids, T* output,
    const T* embed_table, const int max_context_token_num, const int hidden_size) {
    // gid (每个thread负责output的一个元素)
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = index; i < max_context_token_num * hidden_size; 
        i += blockDim.x * gridDim.x) {
        // input: [num_tokens]
        // output: [num_tokens, hidden_size]
        // id: 行号
        // i % hidden_size: 列号
        int id = input_ids[i / hidden_size]; // id: 输入token的id
        output[index] = embed_table[id * hidden_size + i % hidden_size];
    }
}

template<typename T>
void launchInputEmbedding(
    TensorWrapper<int>* input_ids,  // INT  [max_context_token_num] -> 多少个词(token)
    TensorWrapper<T>* output,       // FP32 [max_context_token_num, hidden_size] = [token_num, 4096] -> [多少个词, 每个词用多少维向量表示]
    EmbeddingWeight<T>* embed_tabel // FP32 [vocab_size, hidden_size] -> [词汇表的大小, 每个词用多少维向量表示]
) {
    const int blockNum = 2048;
    const int blockSize  = 256;
    dim3 gird(blockNum);
    dim3 block(blockSize);
    const int max_context_token_num = output->shape[0];
    const int hidden_size           = output->shape[1];

    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], 
        "input ids 1st shape should equal to 1st shape of output");

    // kernel
    embeddingFunctor<T><<<gird, block>>>(
        input_ids->data, output->data, embed_tabel->data, max_context_token_num, hidden_size);
}

// 实例化
template void launchInputEmbedding(TensorWrapper<int>* input_ids, 
                TensorWrapper<float>* output, EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                TensorWrapper<half>* output, EmbeddingWeight<half>* embed_table);
