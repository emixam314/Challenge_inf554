from abc import ABC, abstractmethod
import torch

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
        self.name = f"DL_model({model.name}.{criterion.name}.{optimizer.name})"
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X_train, y_train, epochs, batch_size):
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size].to(self.device)
                y_batch = y_train[i:i+batch_size].to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device)).cpu().numpy()
        