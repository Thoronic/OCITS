import cudf
from cuml.cluster import KMeans
import time

def main():
    input_file = "KMeans_clustering/data.csv"
    k = 5
    expected_num_features = 20
    num_iterations = 20
    output_file = "KMeans_clustering/result.csv"

    print(f"Running KMeans on {input_file} with k={k}")

    df = cudf.read_csv(input_file, dtype='float32', header=None, delimiter=' ')

    # Warm-up the GPU
    df = df.iloc[:, :expected_num_features]
    kmeans = KMeans(n_clusters=k, random_state=1)
    model = kmeans.fit(df)
    cluster_centers = model.cluster_centers_
    cluster_centers.to_csv(output_file)
    
    # Measure the time KMeans on GPU
    time_var = 0
    for _ in range(num_iterations):
        start_time = time.time()
        
        df = df.iloc[:, :expected_num_features]
        kmeans = KMeans(n_clusters=k, random_state=1)
        model = kmeans.fit(df)
        cluster_centers = model.cluster_centers_
        
        end_time = time.time()
        time_var += end_time - start_time
        cluster_centers.to_csv(output_file)

    print("Centers:\n", cluster_centers)
    
    average_time = time_var / num_iterations

    print(f"Average time for kmeans clustering: {average_time} seconds")

if __name__ == "__main__":
    main()
