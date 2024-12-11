from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# Initialiser la barre de progression
tqdm.pandas()

def BERTweet_embedding(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets with BERTweet...")
    # Générer les embeddings
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_bertweet_embeddings(x, tokenizer, model))

    # Convertir les embeddings en colonnes
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    # Ajouter les embeddings au DataFrame
    df = pd.concat([df, embedding_df], axis=1)

    # Préparer le DataFrame final
    period_features = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])

    # Sauvegarder les données dans le dossier embedded_data
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}.csv"
    period_features.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")


def get_bertweet_embeddings(tweet, tokenizer, model):
    # Tokenisation
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    # Utilisation de la dernière couche cachée moyenne comme embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
