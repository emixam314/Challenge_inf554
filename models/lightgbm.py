from lightgbm import LGBMClassifier
from ._Model import Model

class LightGBMClassifier(Model):
    def __init__(self, **model_params):
        """
        Initialise le modèle avec les paramètres donnés.
        Args:
            model_params (dict): Dictionnaire de paramètres pour le modèle scikit-learn.
        """
        self.name = "lightgbm"
        self.model = LGBMClassifier(**model_params)

    def fit(self, X_train, y_train,X_test,y_test):
        """
        Entraîne le modèle sur les données d'entraînement.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Prédit les étiquettes pour les données d'entrée.
        """
        if 'ID' in X.columns:
            X = X.drop(columns=['ID'])

        return self.model.predict(X)