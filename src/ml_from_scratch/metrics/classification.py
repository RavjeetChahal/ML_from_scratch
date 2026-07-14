import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / (len(y_true))

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)
    
def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)
    
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
    
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Predicted labels and true labels must have same length!")
    
    labels = np.unique(np.concatenate([y_true, y_pred]))
    mapping = {}

    for i in range(len(labels)):
        mapping[labels[i]] = i

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        row = mapping[true_label]
        col = mapping[pred_label]
        matrix[row, col] += 1

    return matrix