import pickle
import numpy as np
import pandas as pd
from train_model import decision_tree, classify_example, accuracy
from data_preprocessing import dataset


tree=decision_tree(dataset,100,100)

accuracy = accuracy(dataset,tree)

with open('../models/decision_tree_model_final.pkl','wb') as file:
   pickle.dump(tree, file)


def calculate_metrics(data, tree):
    # Classify each row using the decision tree
    data["classification"] = data.apply(classify_example, axis=1, args=(tree,))
    data[["classification"]].to_csv('../results/train_predictions.csv', index=False)  #Save to CSV
    
    # Create confusion matrix components
    TP = ((data["classification"] == 1) & (data["isFraud"] == 1)).sum()  # True positives
    TN = ((data["classification"] == 0) & (data["isFraud"] == 0)).sum()  # True negatives
    FP = ((data["classification"] == 1) & (data["isFraud"] == 0)).sum()  # False positives
    FN = ((data["classification"] == 0) & (data["isFraud"] == 1)).sum()  # False negatives

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    confusion_matrix = [[TN, FP],[FN, TP]]

    metrics = {
    "Accuracy": accuracy(data, tree),
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1_score,
    "Confusion Matrix": confusion_matrix
    }

    with open('../results/train_metrics.txt', 'w') as file:
        file.write("Classification Metrics:\n")
        file.write(f"Accuracy: {metrics['Accuracy']:.2f}\n")
        file.write(f"Precision: {metrics['Precision']:.2f}\n")
        file.write(f"Recall: {metrics['Recall']:.2f}\n")
        file.write(f"F1-Score: {metrics['F1-Score']:.2f}\n")
        file.write("Confusion Matrix:\n")
        file.write(f"[[{metrics['Confusion Matrix'][0][0]}, {metrics['Confusion Matrix'][0][1]}],\n")
        file.write(f" [{metrics['Confusion Matrix'][1][0]}, {metrics['Confusion Matrix'][1][1]}]]\n")

    return metrics

