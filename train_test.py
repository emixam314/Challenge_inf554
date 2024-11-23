from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd

"""
Ci dessous, le bout à set up pour tester un nouveau model ou preprocessing!
"""

from models.logistic_regression import logistic_regression
from preprocessings.basic_preprocessing import basic_preprocessig

model = logistic_regression
model_name = "logistic_regression"
train_period_features = basic_preprocessig("train_tweets")
eval_period_features = basic_preprocessig("eval_tweets")

""""""

def train_test(model,model_name,train_period_features, eval_period_features):

    # fit et évaluer sur la partie train
    X = train_period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID'])
    y = train_period_features['EventType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test set: ", accuracy_score(y_test, y_pred))

    # obtenir le csv de prédictions à soumettre sur kaggle

    predictions = []
    
    for fname in os.listdir("eval_tweets"):
        
        X = eval_period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

        preds = model.predict(X)

        eval_period_features['EventType'] = preds

        predictions.append(eval_period_features[['ID', 'EventType']])

    pred_df = pd.concat(predictions)
    pred_df.to_csv('data/eval_tweets/outputs/'+model_name+'.csv', index=False)


train_test(model,model_name,train_period_features, eval_period_features)
