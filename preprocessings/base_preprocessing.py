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


def base_preprocessing(raw_folder, processed_folder):

    # Read all training files and concatenate them into one dataframe
    print("concating csv...")
    li = []
    for filename in os.listdir("data/initial_data/" + raw_folder):
        df = pd.read_csv("data/initial_data/" + raw_folder + "/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Apply preprocessing to each tweet
    print("processing text...")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    if not os.path.exists(f"data/processed_data/{processed_folder}"):
        os.makedirs(f"data/processed_data/{processed_folder}")
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"

    df.to_csv(file_path, index=False)



# Basic preprocessing function
def preprocess_text(text):
    ensure_nltk_resources()
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)




