import pandas as pd
from .base_preprocessing import base_preprocessing

def preprocess_data(preprocessing, raw_data_path, processed_data_path):
    if preprocessing == 'base_preprocessing':
        return base_preprocessing(raw_data_path, processed_data_path)
    else:
        raise ValueError(f"Preprocessing method '{preprocessing}' not implemented.")
    

def load_preprocessed_data(processed_folder, raw_folder):
    return pd.read_csv(f"data/processed_data/{processed_folder}/{raw_folder}.csv", index_col=0)