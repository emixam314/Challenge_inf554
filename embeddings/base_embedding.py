import pandas as pd
import numpy as np
import os
import nltk
import gensim.downloader as api


def basic_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model):

    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("transforming in GLOVE...")
    vector_size = 200  # Adjust based on the chosen GloVe model
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # Obtain final dataset: une ligne = labels et output for one minute
    print("preparing final df 1 line = 1 period...")
    period_features = pd.concat([df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    file_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    period_features.to_csv(file_path, index=False)



def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)