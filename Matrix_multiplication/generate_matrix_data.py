import numpy as np

matrix1 = np.random.rand(10000, 10000)
matrix2 = np.random.rand(10000, 10000)

np.save('matrix/matrix1.npy', matrix1)
np.save('matrix/matrix2.npy', matrix2)

print("Matrices saved successfully.")