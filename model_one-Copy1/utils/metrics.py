import numpy as np

def calculate_accuracy(y_pred, y_true):
    y_pred_label = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_label == y_true)