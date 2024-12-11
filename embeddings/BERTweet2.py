from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# Initialiser la barre de progression
tqdm.pandas()

def BERTweet_embedding_word_level(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets into word embeddings with BERTweet...")

    # Générer les embeddings pour chaque token d'un tweet
    df['tweet_words_embeddings'] = df['Tweet'].progress_apply(lambda x: get_word_level_embeddings(x, tokenizer, model))

    # Préparer le DataFrame final
    result_df = df[['ID', 'MatchID', 'PeriodID', 'tweet_words_embeddings']]

    # Sauvegarder les données dans le dossier embedded_data
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Word-level embeddings saved at {output_path}")


def get_word_level_embeddings(tweet, tokenizer, model):
    # Tokenisation
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, return_attention_mask=True)

    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)

    # Récupération des embeddings des tokens (dernière couche cachée)
    # outputs.last_hidden_state a la forme [batch_size, sequence_length, hidden_size]
    token_embeddings = outputs.last_hidden_state.squeeze(0).numpy()

    # Liste des embeddings pour chaque token
    return token_embeddings.tolist()
