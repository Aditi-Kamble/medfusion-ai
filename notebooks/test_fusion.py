import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.app.utils.fusion import MedFusionInference

# Sample patient — you can change these values to test different cases
sample_patient = {
    "age": 58, "sex": 1, "cp": 3, "trestbps": 150, "chol": 283,
    "fbs": 1, "restecg": 0, "thalach": 162, "exang": 0, "oldpeak": 1.0,
    "slope": 2, "ca": 0, "thal": 2
}

# Point this at any real image from your ECG dataset for testing
SAMPLE_ECG_IMAGE = "data/ecg/Normal Person ECG Images (284x12=3408)/Normal(1).jpg"
def main():
    engine = MedFusionInference()
    result = engine.predict_combined(sample_patient, SAMPLE_ECG_IMAGE)

    print("=== MedFusion AI — Risk Assessment ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()