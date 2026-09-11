import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from backend.app.models.ecg_model import ECGNet
from backend.app.utils.explain_ecg import generate_gradcam_overlay

MODEL_PATH = "backend/app/models/ecg_model.pt"

# Adjust this path to match a real image in your project (same one used earlier)
SAMPLE_IMAGE = "data/ecg/Normal Person ECG Images (284x12=3408)/Normal(1).jpg"
SAVE_PATH = "notebooks/gradcam_test_output.jpg"


def main():
    model = ECGNet(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    predicted_class = generate_gradcam_overlay(model, SAMPLE_IMAGE, SAVE_PATH)

    print(f"Predicted class: {predicted_class}")
    print(f"Grad-CAM heatmap saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()