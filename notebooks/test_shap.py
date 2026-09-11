import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from backend.app.models.clinical_model import ClinicalNet
from backend.app.utils.explain_clinical import ClinicalExplainer
from backend.app.utils.preprocess_clinical import load_and_split_data

MODEL_PATH = "backend/app/models/clinical_model.pt"

def main():
    model = ClinicalNet(input_dim=13)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    explainer = ClinicalExplainer(model)

    # Grab one real test patient to explain
    _, X_test, _, y_test = load_and_split_data()
    sample = torch.tensor(X_test[0:1], dtype=torch.float32)

    print(f"Actual label for this test patient: {int(y_test[0])} (1 = disease, 0 = no disease)")

    top_features = explainer.explain(sample, top_k=5)
    print("\nTop contributing factors:")
    for f in top_features:
        print(f"  {f['feature']}: {f['impact']} ({f['direction']})")

if __name__ == "__main__":
    main()