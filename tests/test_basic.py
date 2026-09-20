import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.app.utils.fusion import MedFusionInference

def test_risk_label_valid():
    engine = MedFusionInference()
    sample_patient = {
        "age": 58, "sex": 1, "cp": 3, "trestbps": 150, "chol": 283,
        "fbs": 1, "restecg": 0, "thalach": 162, "exang": 0, "oldpeak": 1.0,
        "slope": 2, "ca": 0, "thal": 2
    }
    result = engine.predict_combined(sample_patient, "data/ecg/Normal Person ECG Images (284x12=3408)/Normal(1).jpg")
    assert result["risk_label"] in ["LOW", "MODERATE", "HIGH"]
    assert 0 <= result["final_risk_score"] <= 100