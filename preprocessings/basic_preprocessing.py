import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def basic_preprocessing(folder,embeddings_model):

    # Read all training files and concatenate them into one dataframe
    print("concating csv...")
    li = []
    for filename in os.listdir("data/initial_data/" + folder):
        df = pd.read_csv("data/initial_data/" + folder + "/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Apply preprocessing to each tweet
    print("processing text...")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    print("transforming in GLOVE...")
    vector_size = 200  # Adjust based on the chosen GloVe model
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # Obtain final dataset: une ligne = labels et output for one minute
    print("preparing final df 1 line = 1 period...")
    period_features = pd.concat([df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    directory = "basic_preprocessing"
    file_path = f"data/processed_data/{directory}/{folder}.csv"
    period_features.to_csv(file_path)



    

def access_basic_processing(folder):

    directory = "basic_preprocessing"
    
    file_path = f"data/processed_data/{directory}/{folder}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable. Le preprocessing n'a peut-être pas encore été calculé, rdv dans _preprocessing_data.py pour faire les calculs.")

    period_features = pd.read_csv(file_path, index_col=0)

    input_size = 200

    return period_features, input_size






  


# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Basic preprocessing function
def preprocess_text(text):
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




