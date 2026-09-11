"""
Trains Branch 1 (ClinicalNet) standalone on the clinical dataset, to verify
it learns something reasonable before we build the fusion model.
Run from the project root: python notebooks/train_clinical.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from backend.app.utils.preprocess_clinical import load_and_split_data
from backend.app.models.clinical_model import ClinicalNet

MODEL_SAVE_PATH = "backend/app/models/clinical_model.pt"
EPOCHS = 100
LR = 0.001
BATCH_SIZE = 16


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    model = ClinicalNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n_samples = X_train_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_preds_binary = (test_preds > 0.5).float()
                acc = accuracy_score(y_test_t.numpy(), test_preds_binary.numpy())
                auc = roc_auc_score(y_test_t.numpy(), test_preds.numpy())
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/n_samples:.4f} "
                  f"| Test Acc: {acc:.4f} | Test AUC: {auc:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()