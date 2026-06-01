"""
credit_fraud_utils_eval.py
Utilities for evaluating and comparing fraud-detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


# Single-model evaluation

def evaluate_model(model, X, y, threshold=0.5):
    """
    Print and return a metrics dictionary for one model on one split.
    threshold is the decision cutoff applied to predicted probabilities.
    Returns a dict with keys: accuracy, precision, recall, f1, auc.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy" : accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall"   : recall_score(y, preds, zero_division=0),
        "f1"       : f1_score(y, preds, zero_division=0),
        "auc"      : roc_auc_score(y, probs),
    }

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y, preds)}")
    print(f"\n  Classification Report:\n{classification_report(y, preds, zero_division=0)}")

    return metrics


# Multi-model comparison

def compare_models(results):
    """
    Print a side-by-side comparison table for multiple models.
    results is a dict of {model_name: metrics_dict} from evaluate_model().
    Returns a DataFrame with one row per model.
    """
    df = pd.DataFrame(results).T
    df = df[["accuracy", "precision", "recall", "f1", "auc"]]
    df.index.name = "Model"

    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(df.to_string(float_format="{:.4f}".format))

    return df


# Threshold tuning

def find_best_threshold(model, X, y):
    """
    Find the decision threshold that maximises F1 on the given split.
    Returns (best_threshold, best_f1).
    """
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-9
    )

    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]

    print(f"\nBest threshold : {best_threshold:.4f}")
    print(f"Best F1-score  : {best_f1:.4f}")

    return best_threshold, best_f1
