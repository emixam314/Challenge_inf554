import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models import FeedforwardNeuralNetModel, LogisticRegression 

config = {
    # General Settings
    'model_type': 'dl',  # Choose between 'ml' or 'dl'

    # Data Paths
    'data_paths': {
        'raw': 'data/raw/raw_data.csv',
        'processed': 'data/processed/',
    },

    # Preprocessing Settings
    'preprocess': True,
    'preprocessing': {
        'lowercase': True,
        'remove_stopwords': True,
        'lemmatize': True,
        'tokenize': True,
        'max_features': 10000,  # Vocabulary size for vectorizers
    },

    # Feature Extraction
    'embeddings': True,
    'features': {
        'method': 'tfidf',  # Options: 'tfidf', 'count_vectorizer', 'word2vec'
    },

    # Deep Learning Model Configurations
    'dl': {
        'model': FeedforwardNeuralNetModel,  # Replace with your actual PyTorch model class
        'model_params': {
            'input_size': 300,  # Example: input size for embeddings
            'hidden_size': 128,  # Example: hidden layer size
            'output_size': 2,  # Number of classes
        },
        'criterion': nn.CrossEntropyLoss,
        'criterion_params': {},  # Add any specific parameters for the criterion
        'optimizer': optim.Adam,
        'optimizer_params': {
            'lr': 0.001,
            'weight_decay': 1e-5,
        },
    },

    # Machine Learning Model Configurations
    'ml': {
        'model' : LogisticRegression,  # Replace with your actual scikit-learn model class
        'model_params': {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'liblinear',
        },
        
    },

    # Training Settings
    'training': {
        'batch_size': 32,
        'epochs': 10,
        'shuffle': True,
    },
}
