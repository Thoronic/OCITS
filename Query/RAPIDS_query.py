import time
import cudf

# Read CSV file into cuDF DataFrame
users = cudf.read_csv("Query/data.csv")
num_iterations = 20

# Warm-up the GPU
result = users[(users['userAge'] >= 50) & (users['userAge'] < 55) & (users['userName'].str.len() >= 12) & (users['termsAccepted'] == True)].sort_values(by='userName')
result.to_csv("Query/result.csv", index=False)

# Record the time for query on GPU
time_var = 0
start_time = time.time()
for i in range(num_iterations):
    start_time = time.time()
    result = users[(users['userAge'] >= 50) & (users['userAge'] < 55) & (users['userName'].str.len() >= 12) & (users['termsAccepted'] == True)].sort_values(by='userName')
    end_time = time.time()
    time_var += end_time - start_time
    result.to_csv("Query/result.csv", index=False)

average_time = time_var / num_iterations

print(result)
print(f"Average time for query on GPU: {average_time} seconds")