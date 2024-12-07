import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from .base_embedding import basic_embedding
from.very_simple_embedding import very_simple_embedding
from .SBERT import SBERT_embedding
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def embedd_data(embedding_type, raw_folder, processed_folder, embedded_folder):
    if embedding_type == 'base_embedding':
        embeddings_model = api.load("glove-twitter-200") 
        embeddings_model = embeddings_model.to(device)
        basic_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
    elif embedding_type == 'very_simple_embedding':
        very_simple_embedding(raw_folder, processed_folder, embedded_folder)
    elif embedding_type == 'SBERT_embedding':
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        SBERT_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)
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