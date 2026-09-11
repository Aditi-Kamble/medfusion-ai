"""
5-fold cross-validation for the clinical model, to report a robust
accuracy/AUC estimate (mean ± std) instead of a single train/test split.

This does NOT replace backend/app/models/clinical_model.pt — it's purely
for generating a stronger evaluation statistic for the report.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from backend.app.models.clinical_model import ClinicalNet
from backend.app.utils.preprocess_clinical import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

EPOCHS = 100
LR = 0.001
BATCH_SIZE = 16
N_FOLDS = 5


def train_one_fold(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.tensor(X_train_scaled)
    y_train_t = torch.tensor(y_train.astype(np.float32)).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled)
    y_test_t = torch.tensor(y_test.astype(np.float32)).unsqueeze(1)

    model = ClinicalNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n_samples = X_train_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_binary = (test_preds > 0.5).float()
        acc = accuracy_score(y_test_t.numpy(), test_preds_binary.numpy())
        auc = roc_auc_score(y_test_t.numpy(), test_preds.numpy())

    return acc, auc


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        acc, auc = train_one_fold(X_train, y_train, X_test, y_test)
        fold_accuracies.append(acc)
        fold_aucs.append(auc)

        print(f"Fold {fold_idx + 1}/{N_FOLDS} — Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    acc_mean, acc_std = np.mean(fold_accuracies), np.std(fold_accuracies)
    auc_mean, auc_std = np.mean(fold_aucs), np.std(fold_aucs)

    print(f"\n=== 5-Fold Cross-Validation Results ===")
    print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/clinical_kfold_results.txt", "w") as f:
        f.write("5-Fold Cross-Validation — Clinical Model\n\n")
        for i, (a, u) in enumerate(zip(fold_accuracies, fold_aucs)):
            f.write(f"Fold {i+1}: Accuracy = {a:.4f}, AUC = {u:.4f}\n")
        f.write(f"\nMean Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%\n")
        f.write(f"Mean AUC: {auc_mean:.4f} ± {auc_std:.4f}\n")

    print("\nSaved to reports/clinical_kfold_results.txt")


if __name__ == "__main__":
    main()