# include <iostream>
# include "input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(const int* input_ids, T* output,
    const T* embed_table, const int max_context_token_num, const int hidden_size)
{
    // 全局thread id(每个token)
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < max_context_token_num * hidden_size)
    {
        // id: 行号(max_context_token_num)
        int id = input_ids[index / hidden_size];

        // id * hidden_size: 所在行的起始地址
        // index % hidden_size: 列索引
        // id * hidden_size + index % hidden_size: 当前token在vocab(词汇表)的地址
        output[index] = embed_table[id * hidden_size + index % hidden_size];

        index += blockDim.x * gridDim.x;
    }
}

template<typename T>
void launchInputEmbedding(
    TensorWrapper<int>* input_ids,  // INT  [max_context_token_num] -> 多少个词(token)
    TensorWrapper<T>* output,       // FP32 [max_context_token_num, hidden_size] = [token_num, 4096] -> [多少个词, 每个词用多少维向量表示]
    EmbeddingWeight<T>* embed_tabel // FP32 [vocab_size, hidden_size] -> [词汇表的大小, 每个词用多少维向量表示]
)
{
    const int blockSize = 256;
    const int gridSize  = 2048;
    const int max_context_token_num = output->shape[0];
    const int hidden_size           = output->shape[1];

    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], 
        "input ids 1st shape should equal to 1st shape of output");

    // kernel
    embeddingFunctor<T><<<gridSize, blockSize>>>(
        input_ids->data, output->data, embed_tabel->data, max_context_token_num, hidden_size);

# ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
# else
# endif
}

// 实例化
template void launchInputEmbedding(TensorWrapper<int>* input_ids, 
                TensorWrapper<float>* output, EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                TensorWrapper<half>* output, EmbeddingWeight<half>* embed_table);
