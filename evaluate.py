from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
def evaluate(y_true, y_pred):
    """
    Evaluate the model using the given true and predicted labels.
    
    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
    """
    
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"F1 score: {f1_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"Confusion matrix: \n{confusion_matrix(y_true, y_pred)}")
    print(f"Classification report: \n{classification_report(y_true, y_pred)}")
    
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)