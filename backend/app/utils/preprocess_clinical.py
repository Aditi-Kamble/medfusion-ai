"""
Preprocessing pipeline for the clinical (tabular) branch of MedFusion AI.
Loads heart.csv, splits into train/test, scales features, and saves the
fitted scaler so the same transformation can be applied at inference time.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/clinical/heart.csv"
SCALER_SAVE_PATH = "backend/app/models/clinical_scaler.pkl"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"


def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler so the FastAPI backend can reuse it later for real patient input
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return (
        X_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        y_train.astype(np.float32),
        y_test.astype(np.float32),
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:", np.bincount(y_train.astype(int)))
    print("Test class balance:", np.bincount(y_test.astype(int)))
    print(f"Scaler saved to {SCALER_SAVE_PATH}")