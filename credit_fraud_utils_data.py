"""
credit_fraud_utils_data.py
Utilities for loading, preprocessing, and resampling credit fraud data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Loading

def load_splits(train_path, val_path, test_path):
    """Load pre-split CSV files and return (train, val, test) DataFrames."""
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    return train, val, test


def print_class_distribution(df, label):
    """Print class distribution for a given split."""
    print(f"\n{label}  (total rows: {df.shape[0]})")
    print(f"  Normal (0): {(df['Class'] == 0).sum()}")
    print(f"  Fraud  (1): {(df['Class'] == 1).sum()}")


# Feature / target splitting

def split_features_target(df):
    """Return (X, y) by dropping the Class column."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


# Scaling

def scale_features(x_train, x_val, x_test):
    """
    Fit a StandardScaler on x_train and transform all three splits.
    Scales the Time and Amount columns.
    Returns (x_train_scaled, x_val_scaled, x_test_scaled, fitted_scaler).
    """
    cols = ["Time", "Amount"]

    scaler  = StandardScaler()
    x_train = x_train.copy()
    x_val   = x_val.copy()
    x_test  = x_test.copy()

    x_train[cols] = scaler.fit_transform(x_train[cols])
    x_val[cols]   = scaler.transform(x_val[cols])
    x_test[cols]  = scaler.transform(x_test[cols])

    return x_train, x_val, x_test, scaler


# Resampling

def apply_smote(X, y):
    """
    Oversample the minority class with SMOTE (random_state=42).
    Returns (X_resampled, y_resampled).
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\nAfter SMOTE:")
    print(y_res.value_counts().to_string())

    return X_res, y_res
