import tensorflow as tf
import time
import pandas as pd

def run_model(features, k, num_iterations, expected_num_features):
    with tf.device("/device:GPU:0"):
        df = features.iloc[:, :expected_num_features]
        
        # Build the model
        input_layer = tf.keras.layers.Input(shape=(df.shape[1],))
        cluster_layer = tf.keras.layers.Dense(k)(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=cluster_layer)
        
        # Compile the model with Mean Squared Error loss
        model.compile(optimizer='adam', loss='mse')
        
        # Fit the model
        target = tf.zeros(((df.shape[0], k)))
        model.fit(df, target, epochs=num_iterations)
        
        cluster_centers = pd.DataFrame(model.layers[1].get_weights()[0])
    return cluster_centers

def main():
    input_file = "KMeans_clustering/data.csv"
    k = 5
    expected_num_features = 20
    num_iterations = 20
    output_file = "KMeans_clustering/result.csv"

    print(f"Running KMeans on {input_file} with k={k}")

    features = pd.read_csv(input_file, header=None, delimiter=' ', dtype='float32')
    
    # Warm-up the CPU
    run_model(features, k, num_iterations, expected_num_features)
    
    # Measure the time KMeans on GPU
    time_start = time.time()
    cluster_centers = run_model(features, k, num_iterations, expected_num_features)
    time_end = time.time()
    
    cluster_centers.to_csv(output_file)
    print("Centers:\n", cluster_centers)
    
    average_time = (time_end - time_start) / num_iterations
    print(f"Average time for kmeans clustering: {average_time} seconds, total time for {num_iterations} epochs: {time_end - time_start} seconds")

if __name__ == "__main__":
    main()
