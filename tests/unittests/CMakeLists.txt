add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(
    rms_norm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnorm)

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
    paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(
    causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(linear
    test_linear.cu
)
target_link_libraries(
    linear PUBLIC
    -lcudart
    -lcudadevrt
    linear_lib) 

add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(
    biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope)

add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
    test_concat_kv PUBLIC
    -lcudart
    -lcudadevrt
    concat_kv)

add_executable(test_repeat_kv
    test_repeat_kv.cu
    )
target_link_libraries(  # Libs for test_qkv_bias_and_rope
    test_repeat_kv PUBLIC
    -lcudart
    -lcudadevrt
    repeat_kv)

add_executable(test_fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  
    test_fused_trans_remv_pad PUBLIC
    -lcudart
    -lcudadevrt
    fused_transpose_and_remv_pad) 