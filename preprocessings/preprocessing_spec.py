import os
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
import nltk

def ensure_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading 'stopwords'...")
        nltk.download('stopwords')

def preprocessing_spec(raw_folder, processed_folder):
    ensure_nltk_resources()

    # Lire tous les fichiers CSV dans le dossier brut
    print("Concatenating CSV files...")
    li = []
    for filename in os.listdir(f"data/initial_data/{raw_folder}"):
        df = pd.read_csv(f"data/initial_data/{raw_folder}/{filename}")
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Appliquer le prétraitement à chaque tweet
    print("Processing tweets...")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    if not os.path.exists(f"data/processed_data/{processed_folder}"):
        os.makedirs(f"data/processed_data/{processed_folder}")
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"

    df.to_csv(file_path, index=False)



def preprocess_text(tweet):
    """
    Prétraiter un tweet pour FastText tout en gardant les éléments spécifiques des tweets.
    """
    # Instancier un tokenizer spécifique aux tweets
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    # Retirer les URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # Tokenisation
    tokens = tokenizer.tokenize(tweet)

    # Garder les hashtags, mentions et emojis, mais retirer la ponctuation inutile
    tokens = [token for token in tokens if token.isalnum() or token.startswith('#') or len(token) == 1 or token in {"!", "!!", "!!!"}]

    # Retourner le tweet reconstruit
    return ' '.join(tokens)


