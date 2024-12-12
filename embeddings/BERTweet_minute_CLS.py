from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# Initialiser la barre de progression
tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def BERTweet_embedding_minute_cls(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets with BERTweet...")
    # Générer les embeddings
    df['embeddings'] = get_bertweet_embeddings_cls_batch(df['Tweet'], tokenizer, model)

    # Convertir les embeddings en colonnes
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    df = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])
    # Ajouter les embeddings au DataFrame
    period_features = pd.concat([df, embedding_df], axis=1)

    # Préparer le DataFrame final
   
    final_df = period_features.groupby('ID')[period_features.iloc[:, 1:].columns].mean().reset_index()

    # Sauvegarder les données dans le dossier embedded_data
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")


def get_bertweet_embeddings_cls_batch(tweets, tokenizer, model, batch_size=32):
    embeddings = []
    print(f"Utilisation de l'appareil : {device}")
    model.to(device)
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), max_length=128, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

