import pandas as pd
import numpy as np
import os

def base_and_very_simple_embedding(raw_folder, processed_folder, embedded_folder):

    file_path = f"data/embedded_data/very_simple_embedding/{raw_folder}.csv"
    very_simple_df = pd.read_csv(file_path)
    file_path = f"data/embedded_data/base_embedding/{raw_folder}.csv"
    base_df = pd.read_csv(file_path)

    if raw_folder == "train_tweets":
        df = pd.merge(very_simple_df, base_df, on=['MatchID', 'PeriodID', 'ID','EventType'], how='outer')

    else:
        df = pd.merge(very_simple_df, base_df, on=['MatchID', 'PeriodID', 'ID'], how='outer')    

    file_path = f"data/embedded_data/{embedded_folder}/{raw_folder}.csv"
    if not os.path.exists(f"data/embedded_data/{embedded_folder}"):
        os.makedirs(f"data/embedded_data/{embedded_folder}")
    df.to_csv(file_path, index=False)