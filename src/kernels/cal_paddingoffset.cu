// 真正的padding是在AddbiasAndPaddingAndRope中进行的
// 此时是在计算偏移, 因为在Attention后需要将padding移除, 所以需要计算偏移
// AddbiasAndPaddingAndRope中进行padding: 在此之前的操作与句子长度无关
# include "src/kernels/cal_paddingoffset.h"

/*
11100
11000
11111
bs = 3
max_q_len = 5
seqlen: [3, 2, 5]
cum_seqlens: 累记句子长度, [0, 3, 5, 10]
paddingoffset: [0, 0, 0, 2, 2, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0]
*/
// 没必要模板化, 因为固定都是int
__global__ void CalPaddingoffset(int* padding_offset, int* cum_seqlens,
    const int* input_lengths, const int batch_size, const int max_q_len)
{
    int index = 0;
    int cum_offset = 0;   // 总偏移
    int total_seqlen = 0; // 总句子长度
    for (int b = 0; b < batch_size; b++) {
        int seqlen = input_lengths[b];  // 获取每个句子的长度(无padding)
        cum_seqlens[b] = total_seqlen;  // 获取总句子长度(无padding)
        for (int i = 0; i < seqlen; ++i) {
            padding_offset[index] = cum_offset; // index: 每个实际的token
            index++;
        }
        cum_offset += (max_q_len - seqlen); // 累记的padding数量
        total_seqlen += seqlen;
    }
    cum_seqlens[batch_size] = total_seqlen;
}

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,
        TensorWrapper<int>* cum_seqlens, TensorWrapper<int>* input_lengths // 实际输入长度
)
{
    const int batch_size = padding_offset->shape[0];
    const int max_q_len = padding_offset->shape[1];
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], "input lenghts numbers should equal to padding offset bs dim!");
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, "cum seqlen numbers should equal to padding offset bs dim + 1!");
    CalPaddingoffset<<<1, 1>>>(padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len);
}