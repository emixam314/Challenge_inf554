from sklearn.linear_model import LogisticRegression
from models._Model import Model

class logistic_regression(Model):

    def __init__(self):
        self.name = "logistic_regression"
        self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)