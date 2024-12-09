import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from .base_embedding import basic_embedding
from.very_simple_embedding import very_simple_embedding
from .SBERT import SBERT_embedding
from .GLOVE_embedding_low_level import embed_tweets_to_sequences
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def embedd_data(embedding_type, raw_folder, processed_folder, embedded_folder):
    if embedding_type == 'base_embedding':
        embeddings_model = api.load("glove-twitter-200")
        basic_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
    elif embedding_type == 'very_simple_embedding':
        very_simple_embedding(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'SBERT_embedding':
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        SBERT_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
    elif embedding_type == 'GLOVE':
        embeddings_model = api.load("glove-twitter-200")
        embed_tweets_to_sequences(raw_folder, processed_folder, embedded_folder, embeddings_model)
    else:
        raise ValueError(f"Embedding method '{embedding_type}' not implemented.")
    
def load_embedded_data(embedded_folder, raw_folder, embedding_type):
    df = pd.read_csv(f"data/embedded_data/{embedded_folder}/{raw_folder}.csv")
    if embedding_type == 'GLOVE':
        X = df.groupby('ID')['tweet_word_embeddings'].apply(list).reset_index()
        X['tweet_word_embeddings'] = X['tweet_word_embeddings'].apply(lambda minute: [tranform(tweet) for tweet in minute])
    else :
        X = df.drop(columns=['MatchID', 'PeriodID'])
    if not 'EventType' in df.columns:
        return X
    X = X.drop(columns=['ID'])
    print('test'+X['tweet_word_embeddings'][0][0], type(X['tweet_word_embeddings'][0][0]))
    y = df.groupby('ID')['EventType'].first().reset_index().drop(columns=['ID'])
    return X, y

def tranform(embedding_str):
    # Step 1: Remove ellipsis (...) and clean the string
    cleaned_str = re.sub(r'\.\.\.|[\[\]]', '', embedding_str).strip()

    # Step 2: Split into rows and convert to a list of floats
    rows = cleaned_str.split('\n')
    embedding_list = [list(map(float, row.split())) for row in rows]

    return embedding_list
