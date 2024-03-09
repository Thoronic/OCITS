import numpy as np
import cupy as cp
import time

num_iterations = 20

# Load matrices from files using NumPy
matrix1 = np.load('Matrix_multiplication/matrix1.npy')
matrix2 = np.load('Matrix_multiplication/matrix2.npy')

# Transfer NumPy arrays to GPU using cuPy
gpu_matrix1 = cp.asarray(matrix1)
gpu_matrix2 = cp.asarray(matrix2)

# Warm-up the GPU
cp.matmul(gpu_matrix1, gpu_matrix2)

# Record the time for matrix multiplication on GPU
start_time = time.time()
for i in range(num_iterations):
    cp.matmul(gpu_matrix1, gpu_matrix2)
end_time = time.time()

# Calculate average time per iteration
average_time = (end_time - start_time) / num_iterations

print(f"Average time for matrix multiplication on GPU (size 10000x10000): {average_time} seconds")
