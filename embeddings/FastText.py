import fasttext
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

tqdm.pandas()

def FastText_embedding(raw_folder, processed_folder, embedded_folder, fasttext_model_path):
    """
    Générer des embeddings pour chaque tweet en utilisant FastText.
    """
    # Charger le modèle FastText pré-entraîné
    print("Loading FastText model...")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    # Charger les tweets prétraités
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming into FastText embeddings...")
    # Générer des embeddings pour chaque tweet
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_fasttext_embeddings(x, fasttext_model))

    # Convertir les embeddings en colonnes séparées
    print("Preparing final DataFrame: one line = one period...")
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    # Fusionner avec le DataFrame original
    df = pd.concat([df, embedding_df], axis=1)

    # Supprimer les colonnes inutiles pour la sortie finale
    period_features = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])

    # Sauvegarder les embeddings
    output_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    period_features.to_csv(output_path, index=False)

    print(f"Saved embeddings to {output_path}")


def get_fasttext_embeddings(tweet, fasttext_model):
    """
    Générer un vecteur d'embedding pour un tweet entier en utilisant FastText.
    Moyenne les embeddings des mots dans le tweet.
    """
    words = tweet.split()  # Découper en mots
    word_embeddings = [fasttext_model.get_word_vector(word) for word in words]
    
    # Moyenne des embeddings des mots
    if word_embeddings:
        tweet_embedding = np.mean(word_embeddings, axis=0)
    else:
        # Si le tweet est vide, renvoyer un vecteur nul
        tweet_embedding = np.zeros(fasttext_model.get_dimension())

    return tweet_embedding
