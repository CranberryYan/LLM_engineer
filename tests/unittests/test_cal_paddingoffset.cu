#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/cal_paddingoffset.h"
// (RussWong)note: this kernel is only int type input and output, not fp32 or half
// we compare the kernel correctnesss by eyes and result print infos
// `./paddingoffset` to run
int main() {
    const int batch_size = 3;
    const int max_q_len = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    int *h_seq_lens;
    int *d_seq_lens;
    h_seq_lens = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_seq_lens, sizeof(int) * batch_size);

    int *h_cum_seqlens;
    int *d_cum_seqlens;
    h_cum_seqlens = (int*)malloc(sizeof(int) * (batch_size + 1));
    cudaMalloc((void**)&d_cum_seqlens, sizeof(int) * (batch_size + 1));
    
    int *h_padding_offset;
    int *d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) * batch_size * max_q_len);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * batch_size * max_q_len);

    /*
    10000
    11000
    11100
    cum_seqlens: [0, 1, 3, 6]
    padding_offset: [0, 4, 4, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    */
    for(int i = 0; i < batch_size; i++) { // 3
       h_seq_lens[i] = i + 1;
    }
    cudaMemcpy(d_seq_lens, h_seq_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    DataType type_int = getTensorType<int>();
    TensorWrapper<int>* padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_q_len}, d_padding_offset);
    TensorWrapper<int>* cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1}, d_cum_seqlens);
    TensorWrapper<int>* input_lengths = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_seq_lens);
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    launchCalPaddingoffset(padding_offset, 
                           cum_seqlens,
                           input_lengths);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_padding_offset, d_padding_offset, sizeof(int) * batch_size * max_q_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, sizeof(int) * (batch_size + 1), cudaMemcpyDeviceToHost);
    // debug info, better to retain: std::cout << "cuda memcpy device to host" << std::endl;    
    printf("padding_offset batch_size: %d\n", padding_offset->shape[0]);
    printf("padding_offset max_q_len:  %d\n", padding_offset->shape[1]);

    for(int i = 0; i < batch_size * max_q_len; i++) {
        printf("padding_offset = %d\n", h_padding_offset[i]);
    }
    for(int i = 0; i < batch_size + 1; i++){
        printf("cum_seqlens =%d\n", h_cum_seqlens[i]);
    }

    free(h_seq_lens);
    free(h_padding_offset);
    free(h_cum_seqlens);
    cudaFree(d_seq_lens);
    cudaFree(d_padding_offset);
    cudaFree(d_cum_seqlens);
}
