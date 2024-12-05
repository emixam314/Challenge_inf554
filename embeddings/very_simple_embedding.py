import pandas as pd
import numpy as np
import os


def very_simple_embedding(raw_folder, processed_folder, embedded_folder):
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    df["count_capital_letter"] = df['Tweet'].apply(count_capital_letter)
    df["count_exclamation"] = df['Tweet'].apply(count_exclamation)
    df["count_tweet_lenght"] = df['Tweet'].apply(count_tweet_lenght)

    df["count_tweet_per_period"] = df.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('count')

    df = df.drop(columns=['Timestamp', 'Tweet'])
    df = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    file_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    df.to_csv(file_path, index=False)



def count_capital_letter(text):
    
    uppercase_count = sum(1 for char in text if char.isupper())
    N = len(text)

    return uppercase_count/N

def count_exclamation(text):

    exclamation_count = text.count('!')
    N = len(text)

    return exclamation_count/N

def count_tweet_lenght(text):
    words = text.split()
    N = len(words)

    return N