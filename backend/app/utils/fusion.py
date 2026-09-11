"""
Inference-time fusion logic for MedFusion AI.

Loads both independently trained models (ClinicalNet, ECGNet) and combines
their outputs for a single patient into one heart disease risk assessment.

IMPORTANT (document this in your report): the two branches were trained on
separate, unrelated datasets (no public paired clinical+ECG dataset exists).
Fusion happens here, at inference time, on a single real patient's actual
concurrent data — not during training. This keeps both branches' reported
accuracy numbers honest and reproducible.
"""

import torch
import pickle
import numpy as np
from PIL import Image

from backend.app.models.clinical_model import ClinicalNet
from backend.app.models.ecg_model import ECGNet
from backend.app.utils.preprocess_ecg import get_train_test_transforms, IDX_TO_CLASS
from backend.app.utils.explain_clinical import ClinicalExplainer
from backend.app.utils.explain_ecg import generate_gradcam_overlay

def generate_recommendation(risk_label, top_features, ecg_class_probs):
    """Builds a detailed, factor-aware recommendation string."""
    increasing = [f["feature"] for f in top_features if f["direction"] == "increases risk"]
    decreasing = [f["feature"] for f in top_features if f["direction"] == "decreases risk"]

    ecg_top_class = max(ecg_class_probs, key=ecg_class_probs.get)
    ecg_top_class_display = ecg_top_class.replace("_", " ")

    lines = []

    if risk_label == "HIGH":
        lines.append(
            "The patient's available data indicates a HIGH estimated risk of heart disease. "
            "Further clinical evaluation by a qualified cardiologist is strongly recommended."
        )
    elif risk_label == "MODERATE":
        lines.append(
            "The patient's available data indicates a MODERATE estimated risk of heart disease. "
            "Clinical follow-up and closer monitoring is recommended."
        )
    else:
        lines.append(
            "The patient's available data indicates a LOW estimated risk of heart disease based on "
            "current inputs. Routine monitoring is recommended."
        )

    if increasing:
        lines.append(
            f"Factors most strongly pushing the risk estimate upward: {', '.join(increasing)}."
        )
    if decreasing:
        lines.append(
            f"Factors helping offset the risk estimate: {', '.join(decreasing)}."
        )

    lines.append(
        f"The submitted ECG was most consistent with the '{ecg_top_class_display}' pattern "
        f"({round(ecg_class_probs[ecg_top_class], 1)}% model confidence)."
    )

    return " ".join(lines)

CLINICAL_MODEL_PATH = "backend/app/models/clinical_model.pt"
CLINICAL_SCALER_PATH = "backend/app/models/clinical_scaler.pkl"
ECG_MODEL_PATH = "backend/app/models/ecg_model.pt"

# Weighting for the final combined risk score.
CLINICAL_WEIGHT = 0.5
ECG_WEIGHT = 0.5

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


class MedFusionInference:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load clinical branch
        self.clinical_model = ClinicalNet(input_dim=len(FEATURE_COLUMNS))
        self.clinical_model.load_state_dict(
            torch.load(CLINICAL_MODEL_PATH, map_location=self.device)
        )
        self.clinical_model.to(self.device).eval()

        with open(CLINICAL_SCALER_PATH, "rb") as f:
            self.clinical_scaler = pickle.load(f)

        self.clinical_explainer = ClinicalExplainer(self.clinical_model)

        # Load ECG branch
        self.ecg_model = ECGNet(num_classes=4)
        self.ecg_model.load_state_dict(
            torch.load(ECG_MODEL_PATH, map_location=self.device)
        )
        self.ecg_model.to(self.device).eval()

        _, self.ecg_transform = get_train_test_transforms()

    def predict_clinical(self, patient_dict):
        """patient_dict must contain all keys in FEATURE_COLUMNS, e.g.:
        {"age": 52, "sex": 1, "cp": 3, "trestbps": 150, "chol": 240,
         "fbs": 1, "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.1,
         "slope": 1, "ca": 1, "thal": 2}
        """
        x = np.array([[patient_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        x_scaled = self.clinical_scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            risk_prob = self.clinical_model(x_tensor).item()

        top_features = self.clinical_explainer.explain(x_tensor, top_k=5)

        return risk_prob, top_features

    def predict_ecg(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.ecg_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.ecg_model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_probs = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}
        ecg_risk = 1.0 - class_probs["Normal"]

        return ecg_risk, class_probs

    def predict_combined(self, patient_dict, ecg_image_path):
        clinical_risk, top_features = self.predict_clinical(patient_dict)
        ecg_risk, ecg_class_probs = self.predict_ecg(ecg_image_path)

        final_risk = CLINICAL_WEIGHT * clinical_risk + ECG_WEIGHT * ecg_risk

        if final_risk >= 0.7:
            risk_label = "HIGH"
        elif final_risk >= 0.4:
            risk_label = "MODERATE"
        else:
            risk_label = "LOW"

        gradcam_path = ecg_image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
        try:
            generate_gradcam_overlay(self.ecg_model, ecg_image_path, gradcam_path)
        except Exception:
            gradcam_path = None
        recommendation = generate_recommendation(risk_label, top_features, ecg_class_probs)

        return {
            "final_risk_score": round(final_risk * 100, 1),
            "risk_label": risk_label,
            "clinical_risk": round(clinical_risk * 100, 1),
            "ecg_risk": round(ecg_risk * 100, 1),
            "ecg_class_probabilities": {k: round(v * 100, 1) for k, v in ecg_class_probs.items()},
            "top_clinical_factors": top_features,
            "gradcam_image_path": gradcam_path,
            "recommendation": recommendation,
        }