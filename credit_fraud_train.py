"""
credit_fraud_train.py
Main entry point for loading data, training models, and evaluating results.

Usage
-----
    python credit_fraud_train.py

Edit the PATHS section below to point to your CSV files before running.
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from credit_fraud_utils_data import (
    load_splits,
    print_class_distribution,
    split_features_target,
    scale_features,
    apply_smote,
)
from credit_fraud_utils_eval import (
    evaluate_model,
    compare_models,
    find_best_threshold,
)


# Paths — update these before running
TRAIN_PATH = "data/split/train.csv"
VAL_PATH   = "data/split/val.csv"
TEST_PATH  = "data/split/test.csv"


# Load data

def load_data():
    print("Loading data...")
    train, val, test = load_splits(TRAIN_PATH, VAL_PATH, TEST_PATH)

    for df, name in [(train, "Train"), (val, "Validation"), (test, "Test")]:
        print_class_distribution(df, name)

    return train, val, test


# Preprocess

def preprocess(train, val, test):
    print("\nPreprocessing...")

    x_train, y_train = split_features_target(train)
    x_val,   y_val   = split_features_target(val)
    x_test,  y_test  = split_features_target(test)

    x_train, x_val, x_test, _ = scale_features(x_train, x_val, x_test)

    x_train_res, y_train_res = apply_smote(x_train, y_train)

    return x_train_res, y_train_res, x_val, y_val, x_test, y_test


# Train individual models

def train_base_models(x_train, y_train):
    print("\nTraining base models...")

    lr_model  = LogisticRegression(max_iter=1000, random_state=42)
    rf_model  = RandomForestClassifier(random_state=42)
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

    lr_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    xgb_model.fit(x_train, y_train)

    return lr_model, rf_model, xgb_model


# Compare base models on validation set

def compare_base_models(lr_model, rf_model, xgb_model, x_val, y_val):
    print("\nEvaluating base models on validation set...")

    results = {}
    for model, name in [
        (lr_model,  "Logistic Regression"),
        (rf_model,  "Random Forest"),
        (xgb_model, "XGBoost"),
    ]:
        print(f"\n{'=' * 45}\n  {name}\n{'=' * 45}")
        results[name] = evaluate_model(model, x_val, y_val)

    compare_models(results)


# Build and tune ensemble

def build_ensemble(x_train, y_train):
    print("\nTraining Voting Classifier ensemble...")

    voting = VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000)),
            ("rf",  RandomForestClassifier(random_state=42)),
            ("xgb", XGBClassifier(eval_metric="logloss")),
        ],
        voting="soft",
    )
    voting.fit(x_train, y_train)
    return voting


def tune_threshold(voting, x_val, y_val):
    print("\nFinding best decision threshold on validation set...")
    best_threshold, _ = find_best_threshold(voting, x_val, y_val)
    return best_threshold


# Save best model

def save_model(voting, best_threshold):
    model_dict = {
        "model":          voting,
        "best_threshold": best_threshold,
        "model_name":     "VotingClassifier (LR + RF + XGB)",
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    print("\nModel saved to model.pkl")


# Final evaluation on test set

def final_evaluation(voting, x_val, y_val, x_test, y_test, best_threshold):
    print("\n\n" + "#" * 55)
    print("  FINAL EVALUATION")
    print("#" * 55)

    print("\n>> Validation set")
    evaluate_model(voting, x_val, y_val, threshold=best_threshold)

    print("\n>> Test set ")
    evaluate_model(voting, x_test, y_test, threshold=best_threshold)


# Main

def main():
    train, val, test = load_data()

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess(train, val, test)

    lr_model, rf_model, xgb_model = train_base_models(x_train, y_train)

    compare_base_models(lr_model, rf_model, xgb_model, x_val, y_val)

    voting = build_ensemble(x_train, y_train)

    best_threshold = tune_threshold(voting, x_val, y_val)

    save_model(voting, best_threshold)

    final_evaluation(voting, x_val, y_val, x_test, y_test, best_threshold)


if __name__ == "__main__":
    main()