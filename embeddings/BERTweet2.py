from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# Initialiser la barre de progression
tqdm.pandas()

import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Fonction pour générer les embeddings avec BERTweet
def get_bertweet_embeddings(tweet, tokenizer, model):
    # Tokenisation
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    # Utilisation de la dernière couche cachée moyenne comme embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Fonction principale pour effectuer le traitement
def BERTweet_embedding_2(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets with BERTweet...")

    # Générer les embeddings pour chaque tweet
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_bertweet_embeddings(x, tokenizer, model))

    # Convertir les embeddings en colonnes (si besoin pour la visualisation)
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    # Ajouter les embeddings au DataFrame
    df = pd.concat([df, embedding_df], axis=1)

    # Convertir le Timestamp en datetime et extraire la minute (0, 1, 2, ...)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['minute'] = df['Timestamp'].dt.minute  # Extraire la minute

    # Calculer la moyenne des embeddings par minute
    df_grouped = df.groupby(['minute', 'MatchID', 'PeriodID']).agg({
        'embeddings': lambda x: np.mean(np.vstack(x), axis=0)  # Moyenne des embeddings par minute
    }).reset_index()

    # Convertir les embeddings en colonnes
    embedding_df_grouped = pd.DataFrame(df_grouped['embeddings'].tolist(), index=df_grouped.index)
    embedding_df_grouped.columns = [f"dim_{i+1}" for i in range(embedding_df_grouped.shape[1])]

    # Ajouter les informations de ID, MatchID, et PeriodID à chaque ligne
    df_grouped = pd.concat([df_grouped[['minute', 'MatchID', 'PeriodID']], embedding_df_grouped], axis=1)
    df_grouped['ID'] = df_grouped.index  # ID unique pour chaque ligne (cela peut être changé si nécessaire)

    # Sauvegarder les données dans le dossier de données embarquées
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}_embedded.csv"
    df_grouped.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")
