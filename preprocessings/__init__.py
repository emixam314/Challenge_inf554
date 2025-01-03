import pandas as pd
from .base_preprocessing import base_preprocessing
from .no_preprocessing import no_preprocessing
from .base_better_preprocessing import base_better_preprocessing


def preprocess_data(preprocessing, raw_data_path, processed_data_path):
    if preprocessing == 'base_preprocessing':
        return base_preprocessing(raw_data_path, processed_data_path)
    elif preprocessing == 'no_preprocessing':
        return no_preprocessing(raw_data_path, processed_data_path)
    elif preprocessing == 'base_better_preprocessing':
        return base_better_preprocessing(raw_data_path, processed_data_path)
    else:
        raise ValueError(f"Preprocessing method '{preprocessing}' not implemented.")
    

def load_preprocessed_data(processed_folder, raw_folder):
    return pd.read_csv(f"data/processed_data/{processed_folder}/{raw_folder}.csv", index_col=0)