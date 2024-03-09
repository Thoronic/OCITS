import tensorflow as tf
import numpy as np
import time
import os

# Set the GPU device(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_iterations = 20

# Load matrices from files
matrix1 = np.load('Matrix_multiplication/matrix1.npy')
matrix2 = np.load('Matrix_multiplication/matrix2.npy')

# Convert NumPy arrays to TensorFlow tensors
tf_matrix1 = tf.constant(matrix1, dtype=tf.float32)
tf_matrix2 = tf.constant(matrix2, dtype=tf.float32)

# Warm-up the GPU
with tf.device("/device:GPU:0"):
    result = tf.matmul(tf_matrix1, tf_matrix2)
    print(result)

# Record the time for matrix multiplication on GPU
time_var = 0
with tf.device("/device:GPU:0"):
    for i in range(num_iterations):
        start_time = time.time()
        result = tf.matmul(tf_matrix1, tf_matrix2)
        end_time = time.time()
        time_var += end_time - start_time
        tf.io.write_file('Matrix_multiplication/result.csv', tf.io.serialize_tensor(result))

# Calculate average time per iteration
average_time = time_var / num_iterations


print(f"Average time for matrix multiplication on GPU (size 10000x10000): {average_time} seconds")