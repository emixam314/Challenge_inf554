import pandas as pd
from .base_preprocessing import base_preprocessing
from .no_preprocessing import no_preprocessing
from .preprocessing_spec import preprocessing_spec


def preprocess_data(preprocessing, raw_data_path, processed_data_path):
    if preprocessing == 'base_preprocessing':
        return base_preprocessing(raw_data_path, processed_data_path)
    elif preprocessing == 'no_preprocessing':
        return no_preprocessing(raw_data_path, processed_data_path)
    elif preprocessing == 'preprocessing_spec':
        return preprocessing_spec(raw_data_path, processed_data_path)
    else:
        raise ValueError(f"Preprocessing method '{preprocessing}' not implemented.")
    

def load_preprocessed_data(processed_folder, raw_folder):
    return pd.read_csv(f"data/processed_data/{processed_folder}/{raw_folder}.csv", index_col=0)