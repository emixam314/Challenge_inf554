import pandas as pd
import gensim.downloader as api
from .base_embedding import basic_embedding
from.very_simple_embedding import very_simple_embedding

def embedd_data(embedding_type, raw_folder, processed_folder, embedded_folder):
    if embedding_type == 'base_embedding':
        embeddings_model = api.load("glove-twitter-200") 
        basic_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model)

    elif embedding_type == 'very_simple_embedding':
        very_simple_embedding(raw_folder, processed_folder, embedded_folder)
        
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