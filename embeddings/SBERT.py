from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Load the pre-trained SBERT model
  # Use a lightweight model for fast processing
tqdm.pandas()


def SBERT_embedding(raw_folder, processed_folder, embedded_folder, embeddings_model):

    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("transforming in SBERT...")
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_sbert_embeddings([x], embeddings_model))

    # Obtain final dataset: une ligne = labels et output for one minute
    print("preparing final df 1 line = 1 period...")
    # Expand embeddings into separate columns
    
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    # Rename columns
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    # Merge with original DataFrame
    df = pd.concat([df, embedding_df], axis=1)

    print(df.head())
    period_features = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])

    file_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    period_features.to_csv(file_path, index=False)



# Generate embeddings for the tweets
def get_sbert_embeddings(tweets, sbert_model):
    return sbert_model.encode(tweets, convert_to_tensor=False)

