from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd

"""
Ci dessous, le bout à set up pour tester un nouveau model ou preprocessing!
"""

from models.logistic_regression import logistic_regression
from preprocessings.basic_and_additionnal_preprocessing import access_basic_and_additionnal_processing

print("preprocessing train_data...")
train_period_features, input_size = access_basic_and_additionnal_processing("sub_train_tweets")
print(input_size)
model = logistic_regression()

""""""

def sub_train_test(model,train_period_features):

    # fit et évaluer sur la partie train
    X = train_period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID'])
    y = train_period_features['EventType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("fitting with train_data...")
    model.fit(X_train, y_train)

    print("testing with train_data...")
    y_pred = model.predict(X_test)
    print("Test set: ", accuracy_score(y_test, y_pred))


sub_train_test(model,train_period_features)
