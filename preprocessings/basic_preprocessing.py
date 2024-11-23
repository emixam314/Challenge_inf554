import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split





def basic_preprocessig(folder):

    # internal import
    nltk.download('stopwords')
    nltk.download('wordnet')
    embeddings_model = api.load("glove-twitter-200")


    # Read all training files and concatenate them into one dataframe
    li = []
    for filename in os.listdir(folder):
        df = pd.read_csv(folder + "/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    # Apply preprocessing to each tweet
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    vector_size = 200  # Adjust based on the chosen GloVe model
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # Obtain final dataset: une ligne = labels et output for one minute
    period_features = pd.concat([df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    return period_features






  


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




