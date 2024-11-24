from abc import ABC, abstractmethod

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
