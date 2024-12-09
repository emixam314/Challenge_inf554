import torch
import os
from preprocessings import preprocess_data, load_preprocessed_data
from embeddings import embedd_data, load_embedded_data
from models import DL_model, LogisticRegression, FeedforwardNeuralNetModel
from config import config
from sklearn.model_selection import train_test_split
from evaluate import evaluate

def main():
    #preprocess data and embeddings (to cook)
    if config['preprocess']:
        print("Preprocessing data...")
        preprocess_data(config['preprocessing_type'], 'train_tweets', config['data_paths']['processed'])
        preprocess_data(config['preprocessing_type'], 'eval_tweets', config['data_paths']['processed'])
        print(f"Data preprocessed and saved in {config['data_paths']['processed']}")

    if config['embeddings']:
        print("Embedding data...")
        embedd_data(config['embedding_type'], 'train_tweets', config['data_paths']['processed'], config['data_paths']['embedded'])
        embedd_data(config['embedding_type'], 'eval_tweets', config['data_paths']['processed'], config['data_paths']['embedded'])
        print(f"Data embedded and saved in {config['data_paths']}")
     
    print(f"Loading data from {config['data_paths']['embedded']}...")
    X, y = load_embedded_data(config['data_paths']['embedded'], 'train_tweets', config['embedding_type'])
    X_valid = load_embedded_data(config['data_paths']['embedded'], 'eval_tweets', config['embedding_type'])

    # Train models
    if config['model_type'] == 'dl':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = config['dl']['criterion'](**config['dl']['criterion_params'])
        nn_model = config['dl']['model'](**config['dl']['model_params'])
        optimizer = config['dl']['optimizer'](nn_model.parameters(), **config['dl']['optimizer_params'])

        model = DL_model(device, nn_model, criterion, optimizer)
    else:
        model = config['ml']['model'](**config['ml']['model_params'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Fitting model...")
    model.fit(X_train, y_train,X_test,y_test, **config['training'])

    # Evaluate models
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

    # Make predictions
    y_valid = model.predict(X_valid)

    # Save predictions
    print("Saving predictions...")
    X_valid['EventType'] = y_valid

    if not os.path.exists('data/valid/'):
        os.makedirs('data/valid/')
    X_valid[['ID', 'EventType']].to_csv(f"data/valid/{config['data_paths']['predictions']}.csv", index=False)



if __name__ == "__main__":
    main()