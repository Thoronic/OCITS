import cudf
from cuml.feature_extraction.text import CountVectorizer
import pandas as pd
import time

def count(df, printing=False):
    start_time = time.time()
    # Tokenize and count words using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])

    # Get the words and their counts
    words = vectorizer.get_feature_names()
    counts = X.sum(axis=0)

    # Create a Pandas DataFrame for easier sorting
    result_df = pd.DataFrame({'word': words.to_numpy(), 'count': counts.get()[0]})
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
    df = cudf.DataFrame({'text': data})
    
    # Warm-up the GPU
    count(df, True)

    time_var = 0
    for _ in range(num_iterations):
        time_var += count(df)
    
    average_time = time_var / num_iterations
    print(f"Average time for counting words: {average_time} seconds")

main()