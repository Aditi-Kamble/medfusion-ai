"""
Generates a per-class precision/recall/F1 breakdown for the ECG model,
to back up the class-imbalance limitation in the report with real numbers.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from sklearn.metrics import classification_report

from backend.app.models.ecg_model import ECGNet
from backend.app.utils.preprocess_ecg import get_dataloaders, IDX_TO_CLASS

MODEL_PATH = "backend/app/models/ecg_model.pt"


def main():
    model = ECGNet(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

    print("Per-Class Performance — ECG Model\n")
    print(report)

    # Also save to a text file for your report
    with open("reports/ecg_per_class_report.txt", "w") as f:
        f.write("Per-Class Performance — ECG Model\n\n")
        f.write(report)

    print("Saved to reports/ecg_per_class_report.txt")


if __name__ == "__main__":
    main()