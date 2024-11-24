import torch
import torch.nn as nn
import torch.optim as optim
from models._Model import Model

class random_neural_network(nn.Module):

    def __init__(self, input_size, lr=0.01):
        
        super(random_neural_network, self).__init__()

        # Définition des couches du modèle
        self.hidden_layer = nn.Linear(input_size, 20)  # Première couche linéaire (entrée -> cachée)
        self.relu = nn.ReLU()                          # Fonction d'activation ReLU
        self.output_layer = nn.Linear(20, 1)           # Deuxième couche linéaire (cachée -> sortie)
        self.sigmoid = nn.Sigmoid()                    # Fonction d'activation Sigmoïde pour classification binaire

        # Fonction de perte pour classification binaire
        self.criterion = nn.BCEWithLogitsLoss()        # BCEWithLogitsLoss combine la sigmoïde et la fonction de perte
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        """
        Définition du passage avant (forward pass).
        Cela correspond à la façon dont les données passent à travers les différentes couches du modèle.
        """

        x = self.hidden_layer(X)

        x = self.relu(x)

        x = self.output_layer(x)

        x = self.sigmoid(x)

        return x  

    def fit(self, X, y, epochs=100):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Redimensionne pour correspondre à la sortie

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            predictions = self(X_tensor)
            
            loss = self.criterion(predictions, y_tensor)
            loss.backward()

            self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            predictions = self(X_tensor).numpy()
            
            # Seuil à 0.5 pour obtenir des prédictions binaires (0 ou 1)
            return (predictions > 0.5).astype(int)  # Si la probabilité est > 0.5, c'est 1, sinon 0
