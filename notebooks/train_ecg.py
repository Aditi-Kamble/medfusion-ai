"""
Trains Branch 2 (ECGNet) standalone on the ECG image dataset.
Run from the project root: python notebooks/train_ecg.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from backend.app.utils.preprocess_ecg import get_dataloaders, IDX_TO_CLASS
from backend.app.models.ecg_model import ECGNet

MODEL_SAVE_PATH = "backend/app/models/ecg_model.pt"
EPOCHS = 30
LR = 0.0001          # lowered from 0.0005 to reduce instability
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def main():
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = ECGNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

        acc, f1 = evaluate(model, test_loader)
        scheduler.step(f1)

        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        marker = "  <- best so far, saved" if is_best else ""
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/n_samples:.4f} "
              f"| Test Acc: {acc:.4f} | Test Macro-F1: {f1:.4f}{marker}")

    print(f"\nBest model saved to {MODEL_SAVE_PATH} (Macro-F1: {best_f1:.4f})")


if __name__ == "__main__":
    main()