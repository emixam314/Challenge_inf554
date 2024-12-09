import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models import FeedforwardNeuralNetModel, LogisticRegression, FeedforwardNeuralNetModelWithDropout, CNNModel, LightGBMClassifier, AttentionModel

config = {
    # General Settings
    'model_type': 'dl',  # Choose between 'ml' or 'dl'

    # Data Paths
    'data_paths': {
        'processed': 'base_processing',
        'embedded': 'GLOVE_embedding',
        'predictions': 'test',
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
    'embedding_type': 'GLOVE',  # Options: 'base_embedding', 'very_simple_embedding', 'SBERT', 'GLOVE'


    'features': {
        'method': 'word2vec()',  # Options: 'tfidf', 'count_vectorizer', 'word2vec'
    },

    # Deep Learning Model Configurations
    'dl': {
        'model': AttentionModel,  # Replace with your actual PyTorch model class
        'model_params': {
            'embedding_dim': 200,  # Example: input size for embeddings
            'hidden_dim': 128,  # Example: hidden size for LSTM
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
        'model' : LightGBMClassifier,  # Replace with your actual scikit-learn model class
        'model_params': {
        },
        
    },

    # Training Settings
    'training': {
        'batch_size':50,
        'epochs': 10,
        'shuffle':False,
    },
}
