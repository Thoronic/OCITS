import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import pandas as pd
import os
import time

# Set the GPU device(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def count(data, printing=False):
    with tf.device("/device:GPU:0"):
        start_time = time.time()
        # Tokenize and count words using TensorFlow's Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)

        # Get the words and their counts
        word_dic = tokenizer.word_counts
        counts = {word: count for word, count in zip(word_dic.keys(), word_dic.values())}

        # Create a Pandas DataFrame for easier sorting
        result_df = pd.DataFrame({'word': list(counts.keys()), 'count': list(counts.values())})
        result_df = result_df.sort_values(by='count', ascending=False)
        end_time = time.time()
        
    if printing:
        print(result_df)
        result_df.to_csv("Word_count/result.csv", index=False)
        
    return end_time - start_time

def main():
    num_iterations = 20
    with open("Word_count/data.txt", 'r', encoding='utf-8') as file:
        data = file.readlines()

    # Warm-up the GPU
    count(data, True)

    time_var = 0
    for _ in range(num_iterations):
        time_var += count(data)
        
    average_time = time_var / num_iterations
    print(f"Average time for counting words: {average_time} seconds")

main()
