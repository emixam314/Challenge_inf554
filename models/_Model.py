from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

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

        if self.model.name == 'attention model':
            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            # Number of samples
            num_samples = len(X_train)

            # Training loop
            for epoch in range(epochs):
                # Shuffle data at the start of each epoch if needed
                indices = np.arange(num_samples)
                if shuffle:
                    np.random.shuffle(indices)

                # Iterate over batches
                for batch_start in range(0, num_samples, batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    batch_samples = [X_train.iloc[idx]['tweet_word_embeddings'] for idx in batch_indices]
                    batch_labels = torch.tensor([y_train.iloc[idx] for idx in batch_indices], dtype=torch.float32).view(-1, 1).to(self.device)
                    # # Prepare hierarchical input
                    # processed_samples = []
                    # for tweets in batch_samples:
                    #     processed_tweets = []
                    #     for tweet in tweets:
                    #         tweet_tensor = torch.tensor(tweet).to(self.device)  # Shape: (num_words, embedding_dim)
                    #         processed_tweets.append(tweet_tensor)
                    #     processed_samples.append(processed_tweets)

                    # Training mode
                    self.model.train()

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    y_pred = self.model(batch_samples)  # Model handles hierarchical inputs

                    # Compute loss
                    loss = self.criterion(y_pred, batch_labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    # Optional: Logging progress
                    print(f"Epoch {epoch + 1}, Batch {batch_start // batch_size + 1}, Loss: {loss.item()}")


        else :
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

                        accuracy = 100. * correct / total

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