
Welcome to the project repository for predicting notable events during football matches using Twitter data. The goal of the project is to predict, based on tweet data, whether a given one-minute interval from a football match corresponds to a notable event such as a goal, penalty, substitution, etc.

This README will guide you on how to set up, run, and experiment with the project.


# Project Overview

The project uses tweets from football matches (2010 and 2014 FIFA World Cups) to predict the occurrence of notable events during specific one-minute intervals. The task is to classify whether a tweet corresponds to a notable event such as goals, substitutions, red cards, etc.

The core of the project is based on:
- Data Preprocessing
- Text Embedding using models like BERTweet
- Classification using machine learning models (LSTM, NN, LightGBM)


## Installation

### Prerequisites

To run the project, you will need Python 3.7 or above. Additionally, the project requires several Python libraries, which can be installed via `pip`.

Install the dependencies:

   pip install -r requirements.txt

   /!\ one line is commented, you have to install it manually

Create a "Data" folder as follows : 

Data/
|__  initial_data/
        |__  train_tweets/          # insert the csv files here
        |__  eval_tweets/           # insert the csv files here


## Directory Structure

The project is structured as follows:


CHALLENG_INF_554/
│
├── data/                               # Data files
│   ├── initial_data/       
│   ├── embedded_data/        
│   └── preocessed_data/  
|
|
├── embeddings/                         # Contains scripts for data embedding (BERTweet, GloVe, etc.)
|   |__ __init__.py                     # Contains scripts to run the embedding
|   |__ base_emebedding.py              # Embedding of the baseline
│   |__ BERTweet_minute_batch_gpu.py    # Main script for embedding tweets using BERTweet, gives one vector per minute
│   |__ BERTweet_minute_CLS.py          # Script for embedding tweets using BERTweet and CLS
|   |__ BERTweet.py                     # First version of BERTweet, gives one vector per tweet
│   |__SBERT.py                         # Script for embedding tweets using SBERT
|   |__ very_simple_embedding.py        # Script for basic approach
│
├── models/                             # Machine learning models and training scripts
|   |__ __init__.py                     # Contains scripts to run the models
│   |__ NN_model/
|   |       |__ __init__.py
|   |       |__ attention_model.py      # Not usable
|   |       |__ CNN.py
|   |       |__ feed_forward_nerual_net.py
|   |       |__ feed_forward_with_dropout.py
|   |       |__ NN_Model.py
│   |__ LGBM.py       
│   |__ logistic_regression.py          # Random Neural Network for classification
│
├── preprocessing/                      # Data preprocessing scripts
|   |__ __init__.py                     # Contains scripts to run the preprocessings
│   |__ base_better_preprocessing.py    # Preprocessing of the baseline, improved
│   |__ base_preprocessing.py           # Preprocessing of the baseline
│   |__ no_preprocessing.py             # For basic approach
│
├── config.py                           # Configuration file for various settings           
│
├── main.py                             # Main script to run the entire pipeline
├── requirements.txt                    # Python dependencies
└── README.md                           # This file


## Configuration

Before running the project, you can configure the parameters and options through the `config/config.py` file.


- Preprocessing Type: Choose between 
    - no_preprocessing
    - base_preprocessins
    - base_better_preprocessing

- Embedding Type: Choose between 
    - base_embedding
    - very_simple_embedding
    - SBERT
    - BERTweet
    - BERTweet_minute
    - BERTweet_minute_cls
    - BERTweet_minute_batch
- Model Type: Choose between

- Data Paths: Adjust the paths for processed data and embedded data.

For each part, choose between True or False to run this part or not.

## Running the Project

To run the entire pipeline, use the `main.py` script. This script will automatically run preprocessing, embedding, and model training/evaluation according to yout configuration.


python main.py

