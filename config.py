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
        'processed': 'base_better_preprocessing',
        'embedded': 'BERTweet_minute_batch',
        'predictions': 'base_predictions',
    },

    # Preprocessing Settings
    'preprocess': False,
    'preprocessing_type': 'base_better_preprocessing',

   
    # Embedding Settings
    'embeddings': True,
    'embedding_type': 'BERTweet_minute_batch',


    # Deep Learning Model Configurations
    'dl': {
        'model': FeedforwardNeuralNetModel,  # Replace with the model specified
        'model_params': {
            'input_dim': 200,  # Example: input size for embeddings
            'hidden_dim': 128,  # Example: hidden layer size
            'output_dim': 1,  # Number of classes
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
        'model' : LogisticRegression,  # Replace with the model specified
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
