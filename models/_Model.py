from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class Model(ABC):

    def __init__(self):
        if not hasattr(self, 'name'):
            raise NotImplementedError("Chaque modèle doit avoir un attribut 'name'.")

    @abstractmethod
    def fit(self, X_train, y_train,X_test,y_test):
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

    def fit(self, X_train, y_train,X_test,y_test, epochs, batch_size, shuffle):

        trainset = TensorDataset(torch.from_numpy(X_train.values).float().to(self.device), torch.from_numpy(y_train.values).float().to(self.device))
        train_loader = DataLoader(trainset, shuffle=shuffle, batch_size=batch_size)
        testset = TensorDataset(torch.from_numpy(X_test.values).float().to(self.device), torch.from_numpy(y_test.values).float().to(self.device))
        test_loader = DataLoader(testset, shuffle=shuffle, batch_size=batch_size)

        iter = 0

        history_train_acc, history_val_acc, history_train_loss, history_val_loss = [], [], [], []

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

                iter += 1

                if iter % 100 == 0:
                    # Get training statistics
                    train_loss = loss.data.item()

                    # Testing mode
                    self.model.eval()
                    # Calculate Accuracy         
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for samples, labels in test_loader:
                        # Load samples
                        samples = samples.view(-1, self.model.input_dim).to(self.device)
                        labels = labels.view(-1, 1).to(self.device)

                        # Forward pass only to get logits/output
                        predicted = self.model(samples)

                        # Val loss
                        val_loss = self.criterion(predicted, labels)

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted.type(torch.FloatTensor).cpu() == labels.type(torch.FloatTensor)).sum().item()
                        # correct = (predicted == labels.byte()).int().sum().item()

                    accuracy = correct / total

                    history_val_loss.append(val_loss.data.item())
                    history_val_acc.append(round(accuracy, 2))
                    history_train_loss.append(train_loss)

        self.plot_losses(history_train_loss, history_val_loss)


    def predict(self, X):
        if 'ID' in X.columns:
            X = X.drop(columns=['ID'])
        X = torch.tensor(X.values, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X.to(self.device)).cpu().numpy()

            # Seuil à 0.5 pour obtenir des prédictions binaires (0 ou 1)
            return (predictions > 0.5).astype(int)
        
    def plot_losses(self,history_train_loss, history_val_loss):
        # Set plotting style
        #plt.style.use(('dark_background', 'bmh'))
        plt.style.use('bmh')
        plt.rc('axes', facecolor='none')
        plt.rc('figure', figsize=(16, 4))

        # Plotting loss graph
        plt.plot(history_train_loss, label='Train')
        plt.plot(history_val_loss, label='Validation')
        plt.title('Loss Graph')
        plt.legend()
        
        plt.savefig("plot_losses", dpi=300)  # Save with a high resolution
        plt.close()

    def plot_accuracy(self,history_val_acc):
        # Set plotting style
        #plt.style.use(('dark_background', 'bmh'))
        plt.style.use('bmh')
        plt.rc('axes', facecolor='none')
        plt.rc('figure', figsize=(16, 4))

        # Plotting loss graph
        plt.plot(history_val_acc, label='history_val_acc')
        plt.title('Accuracy Graph')
        plt.legend()
        
        plt.savefig("plot_losses", dpi=300)  # Save with a high resolution
        plt.close()