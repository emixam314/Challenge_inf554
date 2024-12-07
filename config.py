import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models import FeedforwardNeuralNetModel, LogisticRegression, FeedforwardNeuralNetModelWithDropout, CNNModel

config = {
    # General Settings
    'model_type': 'dl',  # Choose between 'ml' or 'dl'

    # Data Paths
    'data_paths': {
        'processed': 'base_processing',
        'embedded': 'base_embedding',
        'predictions': 'base_predictions_CNN',
    },

    # Preprocessing Settings (note used for now)
    'preprocess': False,
    'preprocessing_type': 'base_preprocessing',

    'preprocessing': {
        'lowercase': True,
        'remove_stopwords': True,
        'lemmatize': True,
        'tokenize': True,
        'max_features': 10000,  # Vocabulary size for vectorizers
    },

    # Feature Extraction (not used for now)
    'embeddings': False,
    'embedding_type': 'SBERT_embedding',


    'features': {
        'method': 'word2vec()',  # Options: 'tfidf', 'count_vectorizer', 'word2vec'
    },

    # Deep Learning Model Configurations
    'dl': {
        'model': CNNModel,  # Replace with your actual PyTorch model class
        'model_params': {
            'input_dim': 200,  # Example: input size for embeddings
            'num_classes': 1,  # Number of classes
        },
        'criterion': nn.BCELoss,
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
        'batch_size':32,
        'epochs': 100,
        'shuffle':True,
    },
}
