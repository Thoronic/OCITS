import pandas as pd
import tensorflow as tf
import os
import time

def main():
    # Set the GPU device(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_iterations = 20
    users = pd.read_csv("Query/data.csv")

    # Convert pandas DataFrame to TensorFlow Dataset
    users_dataset = tf.data.experimental.SqlDataset.from_tensor_slices(users.to_dict('list'))

    # Warm-up the GPU
    with tf.device("/device:GPU:0"):
        # Filtering ("WHERE")
        filtered = users_dataset.filter(lambda x: x['userAge'] >= 50 and x['userAge'] < 55 and tf.strings.length(x['userName']) >= 12 and x['termsAccepted'])

        # Sorting ("ORDER BY") (converting tf SQLDataset to pandas DataFrame for sorting)
        df = pd.DataFrame(list(filtered.as_numpy_iterator()))
        sorted = df.sort_values(by='userName')
    
    
    # Record the time for matrix multiplication on GPU
    time_where = 0
    time_both = 0
    with tf.device("/device:GPU:0"):
        for i in range(num_iterations):
            start_time = time.time()
            
            filtered = users_dataset.filter(lambda x: x['userAge'] >= 50 and x['userAge'] < 55 and tf.strings.length(x['userName']) >= 12 and x['termsAccepted'])
            end_time_where = time.time()
            
            df = pd.DataFrame(list(filtered.as_numpy_iterator()))
            sorted = df.sort_values(by='userName')
            end_time_both = time.time()
            
            time_where += end_time_where - start_time
            time_both += end_time_both - start_time
            

    # Calculate average time per iteration
    average_time_where = time_where / num_iterations
    average_time_both = time_both / num_iterations
    
    # Show the result
    print(sorted)

    # Save the result to CSV
    sorted.to_csv("Query/result.csv", index=False)

    print(f"Average time for SQL query (only WHERE query): {average_time_where} seconds")
    print(f"Average time for SQL query (both queries): {average_time_both} seconds")

main()
