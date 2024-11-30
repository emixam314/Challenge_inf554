import torch
from models import DL_model, LogisticRegression, FeedforwardNeuralNetModel
from config import config

def main():
    #preprocess data and embeddings (to cook)
    if config['preprocess']:
        print("Preprocessing data...")
        X_train, y_train, X_valid = preprocess_data(config['preprocessing'], config['data_paths']['raw'], config['data_paths']['processed'])
        print(f"Data preprocessed and saved in {config['data_path']}")
    elif config['embeddings']:
        print(f"Loading data from {config['data_path']}...")
        X_train, y_train, X_valid = load_data(config['data_paths']['processed'])

    # Train models
    if config['model_type'] == 'dl':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = config['dl']['criterion'](config['dl']['criterion_params'])
        optimizer = config['dl']['optimizer'](config['dl']['optimizer_params'])
        nn_model = config['dl']['model'](config['dl']['model_params'])

        model = DL_model(device, nn_model, criterion, optimizer)
    else:
        model = config['ml']['model'](config['ml']['model_params'])
    
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Evaluate models
    #TODO: Implement evaluate method in models
    model.evaluate()

    # Make predictions
    y_pred = model.predict(X_valid)


if __name__ == "__main__":
    main()