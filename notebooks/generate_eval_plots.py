"""
Generates confusion matrix + ROC curve plots for both models, saved to
reports/ for use in your project report/presentation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

from backend.app.models.clinical_model import ClinicalNet
from backend.app.utils.preprocess_clinical import load_and_split_data
from backend.app.models.ecg_model import ECGNet
from backend.app.utils.preprocess_ecg import get_dataloaders, IDX_TO_CLASS

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def clinical_plots():
    model = ClinicalNet(input_dim=13)
    model.load_state_dict(torch.load("backend/app/models/clinical_model.pt", map_location="cpu"))
    model.eval()

    _, X_test, _, y_test = load_and_split_data()
    X_test_t = torch.tensor(X_test)

    with torch.no_grad():
        probs = model(X_test_t).numpy().flatten()
    preds = (probs > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test.astype(int), preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Clinical Model — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "clinical_confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test.astype(int), probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#1a3b5d", label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Clinical Model — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "clinical_roc_curve.png"), dpi=150)
    plt.close()

    print("Clinical model plots saved.")


def ecg_plots():
    model = ECGNet(num_classes=4)
    model.load_state_dict(torch.load("backend/app/models/ecg_model.pt", map_location="cpu"))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=16)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    class_names = [IDX_TO_CLASS[i] for i in range(4)]
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("ECG Model — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "ecg_confusion_matrix.png"), dpi=150)
    plt.close()

    print("ECG model plots saved.")


if __name__ == "__main__":
    clinical_plots()
    ecg_plots()
    print(f"\nAll evaluation plots saved to {REPORTS_DIR}/")