from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader

class Model(ABC):

    def __init__(self):
        if not hasattr(self, 'name'):
            raise NotImplementedError("Chaque modèle doit avoir un attribut 'name'.")

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        X_train et y_train sont des dataframes
        """
        pass

    @abstractmethod
    def predict(self, X):
        pass

class DL_model(Model):
    
    def __init__(self, device, model, criterion, optimizer):
        self.name = f"DL_model({model.name})"
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X_train, y_train, epochs, batch_size, shuffle):
        trainset = TensorDataset(torch.from_numpy(X_train.values).float().to(self.device), torch.from_numpy(y_train.values).float().to(self.device))
        train_loader = DataLoader(trainset, shuffle=shuffle, batch_size=batch_size)
        for epoch in range(epochs):
            for i, (samples, labels) in enumerate(train_loader):
                # Training mode
                self.model.train()

                # Load samples
                samples = samples.view(-1, self.model.input_dim).to(self.device)
                labels = labels.view(-1, 1).to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(samples)
                loss = self.criterion(y_pred, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        if 'ID' in X.columns:
            X = X.drop(columns=['ID'])
        X = torch.tensor(X.values, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X.to(self.device)).cpu().numpy()

            # Seuil à 0.5 pour obtenir des prédictions binaires (0 ou 1)
            return (predictions > 0.5).astype(int)