import numpy as np
import pandas as pd
from tqdm import tqdm
import os

best_max_words = 0
# Embed words for all tweets in the dataset
def embed_tweets_to_sequences(raw_folder, processed_folder, embedded_folder, embeddings_model, max_words=None):

    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    global best_max_words
    
    # Embed tweets into sequences of word embeddings
    tqdm.pandas()  # Enable progress bar
    df['tweet_word_embeddings'] = df['Tweet'].progress_apply(
        lambda tweet: embed_tweet_words(tweet, embeddings_model, embedding_dim=200, max_words=max_words)
    )
    df = df.drop(columns=['Timestamp', 'Tweet'])

    print(best_max_words)
    file_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    df.to_csv(file_path, index=False)


# Function to embed a single tweet into a sequence of word embeddings
def embed_tweet_words(tweet, embedding_model, embedding_dim=200, max_words=None):
    global best_max_words
    tokens = tweet.lower().split()  # Tokenize the tweet
    embeddings = [embedding_model[word] for word in tokens if word in embedding_model]
    
    # Pad or truncate to max_words
    if max_words:
        if len(embeddings) < max_words:
            embeddings += [np.zeros(embedding_dim)] * (max_words - len(embeddings))
        else:
            embeddings = embeddings[:max_words]
    else :
        best_max_words = len(embeddings) if len(embeddings) > best_max_words else best_max_words
    
    return np.array(embeddings)  # Shape: (max_words, embedding_dim)
