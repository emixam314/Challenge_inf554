import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk


def ensure_nltk_resources():
    try:
        # Check if 'stopwords' is already downloaded
        stopwords.words('english')
    except LookupError:
        print("Downloading 'stopwords'...")
        nltk.download('stopwords')

    try:
        # Check if 'wordnet' is already downloaded
        wordnet.synsets('example')
    except LookupError:
        print("Downloading 'wordnet'...")
        nltk.download('wordnet')


def base_better_preprocessing(raw_folder, processed_folder):
    """
    This function processes tweets by removing RT, @mentions, and keeping punctuation and emojis.
    It also removes stopwords and lemmatizes the text.
    """

    # Read all training files and concatenate them into one dataframe
    print("concating csv...")
    li = []
    for filename in os.listdir("data/initial_data/" + raw_folder):
        df = pd.read_csv("data/initial_data/" + raw_folder + "/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Apply preprocessing to each tweet
    print("Processing text...")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    if not os.path.exists(f"data/processed_data/{processed_folder}"):
        os.makedirs(f"data/processed_data/{processed_folder}")
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    
    df.to_csv(file_path, index=False)


def preprocess_text(text):
    """
    Preprocess text by removing RT, @mentions, stopwords, lemmatizing words, 
    but keeping emojis and punctuation.
    """

    ensure_nltk_resources()

    # Lowercase the text
    text = text.lower()

    # Enlever les liens
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove 'RT' (retweet indicator) and mentions (i.e., @username)
    text = re.sub(r'\bRT\b', '', text)  # Remove "RT" (Retweet indicator)
    text = re.sub(r'@\w+:?', '', text)  # Enlève @mention et le ":" suivant si présent

    # Enlever les caractères qui ne sont pas des mots, espaces, ou ponctuation (garde les chiffres)
    #text = re.sub(r'[^\w\s.,!?\'";:()&-]', '', text)

    # Remove "|"
    text = re.sub(r'\|', '', text)

    # Tokenization: Split by whitespace
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Rejoin the text
    return ' '.join(words)

