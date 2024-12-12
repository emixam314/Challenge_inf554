import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

tqdm.pandas()

def BERTweet_embedding(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets with BERTweet...")
    # Générer les embeddings pour chaque tweet
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_bertweet_embeddings(x, tokenizer, model))

    # Regrouper par minute (PeriodID) et calculer l'embedding moyen pour chaque minute
    grouped = (
        df.groupby(['ID', 'MatchID', 'PeriodID'], as_index=False)
        .agg({'embeddings': lambda x: torch.mean(torch.tensor(x.tolist()), dim=0).numpy()})
    )

    # Sauvegarder les données dans le dossier embedded_data
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}_minute_embeddings.csv"
    grouped.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")

def get_bertweet_embeddings(tweet, tokenizer, model):
    # Tokenisation
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    # Utilisation de la dernière couche cachée moyenne comme embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
