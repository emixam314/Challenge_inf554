import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from .base_embedding import basic_embedding
from.very_simple_embedding import very_simple_embedding
from .SBERT import SBERT_embedding
import torch
from .BERTweet import BERTweet_embedding
from .BERTweet_minute import BERTweet_embedding_minute
from .BERTweet_minute_CLS import BERTweet_embedding_minute_cls
from .BERTweet_minute_batch_gpu import BERTweet_embedding_minute


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedd_data(embedding_type, raw_folder, processed_folder, embedded_folder):
    if embedding_type == 'base_embedding':
        embeddings_model = api.load("glove-twitter-200") 
        embeddings_model = embeddings_model.to(device)
        basic_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
    elif embedding_type == 'very_simple_embedding':
        very_simple_embedding(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'SBERT':
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        SBERT_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
    elif embedding_type == 'BERTweet': 
        BERTweet_embedding(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'BERTweet_minute' :
        BERTweet_embedding_minute(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'BERTweet_minute_cls' :
        BERTweet_embedding_minute_cls(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'BERTweet_minute_batch':
        BERTweet_embedding_minute(raw_folder, processed_folder, embedded_folder)
    else:
        raise ValueError(f"Embedding method '{embedding_type}' not implemented.")
    
def load_embedded_data(embedded_folder, raw_folder):
    df = pd.read_csv(f"data/embedded_data/{embedded_folder}/{raw_folder}.csv")
    X = df.drop(columns=['MatchID', 'PeriodID'])
    if not 'EventType' in df.columns:
        return X
    X = X.drop(columns=['EventType','ID'])
    y = df['EventType']
    return X, y