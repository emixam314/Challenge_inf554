from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd

"""
Ci dessous, le bout à set up pour tester un nouveau model ou preprocessing!
"""
from models.random_neural_network import random_neural_network
from preprocessings.basic_preprocessing import access_basic_processing

print("preprocessing train_data...")
train_period_features, input_size = access_basic_processing("sub_train_tweets")
print("preprocessing test_data...")
eval_period_features, input_size = access_basic_processing("eval_tweets")

model = random_neural_network(input_size=input_size)

""""""

def train_test(model,train_period_features, eval_period_features):

    # fit et évaluer sur la partie train
    X = train_period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID'])
    y = train_period_features['EventType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("fitting with train_data...")
    model.fit(X_train, y_train)

    print("testing with test_data...")
    y_pred = model.predict(X_test)
    print("Test set: ", accuracy_score(y_test, y_pred))

    # obtenir le csv de prédictions à soumettre sur kaggle
    print("predicting eval_data...")
    predictions = []
    for fname in os.listdir("data/eval_tweets"):
        
        X = eval_period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

        preds = model.predict(X)

        eval_period_features['EventType'] = preds

        predictions.append(eval_period_features[['ID', 'EventType']])

    pred_df = pd.concat(predictions)
    pred_df.to_csv('data/outputs/'+model.name+'.csv', index=False)


train_test(model,train_period_features, eval_period_features)
