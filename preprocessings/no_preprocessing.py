import os
import re
import numpy as np
import pandas as pd


def no_preprocessing(raw_folder, processed_folder):

    # Read all training files and concatenate them into one dataframe
    print("concating csv...")
    li = []
    for filename in os.listdir("data/initial_data/" + raw_folder):
        df = pd.read_csv("data/initial_data/" + raw_folder + "/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)

    if not os.path.exists(f"data/processed_data/{processed_folder}"):
        os.makedirs(f"data/processed_data/{processed_folder}")
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"

    df.to_csv(file_path, index=False)


