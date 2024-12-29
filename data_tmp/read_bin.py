import numpy as np

def read_tensor(filename, shape, dtype=np.float32):
    # Load binary data
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    # Reshape to the original tensor dimensions
    tensor = data.reshape(shape)
    return tensor

# Example Usage
filename = "/home/yst/文档/yst/LLM/熊猫老师/my_LLM_engineering/data_tmp/0_qk_v_buf_after_bmm.bin"
Bm, Bk = 8, 64  # Replace with the correct dimensions
tensor = read_tensor(filename, shape=(Bm, Bk))

# Print some elements
print(tensor)