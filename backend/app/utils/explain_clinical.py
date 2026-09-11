"""
SHAP-based explainability for the clinical branch (ClinicalNet).
Explains which clinical features drove a specific patient's risk prediction.
"""

import shap
import torch
import numpy as np

from backend.app.utils.preprocess_clinical import load_and_split_data, FEATURE_COLUMNS

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels (ca)",
    "thal": "Thalassemia",
}


class ClinicalExplainer:
    def __init__(self, model, background_size=50):
        """
        model: a loaded ClinicalNet instance (already in eval mode)
        """
        self.model = model
        X_train, _, _, _ = load_and_split_data()
        background_idx = np.random.choice(X_train.shape[0], size=min(background_size, X_train.shape[0]), replace=False)
        self.background = torch.tensor(X_train[background_idx], dtype=torch.float32)

        # DeepExplainer expects the model to return a single output tensor per call
        self.explainer = shap.DeepExplainer(self.model, self.background)

    def explain(self, x_scaled_tensor, top_k=5):
        """
        x_scaled_tensor: a (1, 13) already-scaled input tensor for ONE patient
        Returns a list of (feature_display_name, shap_value) sorted by |impact|, top_k entries.
        """
        shap_values = self.explainer.shap_values(x_scaled_tensor, check_additivity=False)

        # shap_values shape: (1, 13, 1) or (1, 13) depending on shap version — normalize to flat array
        values = np.array(shap_values).reshape(-1)

        feature_impacts = list(zip(FEATURE_COLUMNS, values))
        feature_impacts.sort(key=lambda pair: abs(pair[1]), reverse=True)

        top_features = [
            {
                "feature": FEATURE_DISPLAY_NAMES.get(name, name),
                "impact": round(float(val), 4),
                "direction": "increases risk" if val > 0 else "decreases risk",
            }
            for name, val in feature_impacts[:top_k]
        ]
        return top_features