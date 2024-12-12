import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from joblib import Parallel, delayed

def get_bertweet_embeddings_batch(tweets, tokenizer, model, device):
    # Valider les entrées : vérifier que c'est une liste et que tous les éléments sont des chaînes
    if not isinstance(tweets, list):
        raise ValueError("Input `tweets` must be a list of strings.")
    
    tweets = [tweet for tweet in tweets if isinstance(tweet, str)]
    if not tweets:  # Si la liste est vide après le filtrage
        raise ValueError("All elements in `tweets` must be strings. Received an empty or invalid batch.")
    
    # Tokenisation
    inputs = tokenizer(tweets, max_length=128, return_tensors="pt", truncation=True, padding=True)

    # Déplacer les tenseurs sur l'appareil spécifié (GPU ou CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Récupérer les embeddings à partir de la dernière couche cachée
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings



def BERTweet_embedding_minute(raw_folder, processed_folder, embedded_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)

    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    # Nettoyage des données
    df = df.dropna(subset=['Tweet'])  # Retirer les lignes avec des valeurs NaN dans `Tweet`
    df['Tweet'] = df['Tweet'].astype(str)  # Convertir toutes les valeurs en chaînes

    print("Transforming tweets with BERTweet...")
    batch_size = 32
    embeddings = []

    for i in range(0, len(df), batch_size):
        batch = df['Tweet'].iloc[i:i + batch_size].tolist()
        # Filtrer uniquement les chaînes valides
        batch = [tweet for tweet in batch if isinstance(tweet, str)]
        if not batch:
            print(f"Skipping an empty or invalid batch at index {i}")
            continue

        embeddings_batch = get_bertweet_embeddings_batch(batch, tokenizer, model, device)
        embeddings.extend(embeddings_batch)

    embedding_df = pd.DataFrame(embeddings, index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]
    df = pd.concat([df, embedding_df], axis=1)

    period_features = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])
    final_df = period_features.groupby('ID')[period_features.iloc[:, 1:].columns].mean().reset_index()

    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")

