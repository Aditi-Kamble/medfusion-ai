"""
MedFusion AI — FastAPI backend entry point.
Run from project root: uvicorn backend.app.main:app --reload
"""

import os
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

from backend.app.database import engine, get_db, Base
from backend.app.models_db import Doctor, Patient, Assessment
from backend.app.utils.fusion import MedFusionInference
from backend.app.routes.frontend import router as frontend_router
from backend.app.routes.patient import router as patient_router

# Create all tables on startup (safe to call repeatedly — no-op if they exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="MedFusion", description="Multi-Modal Heart Disease Risk Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a college project demo; restrict in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: secret_key is hardcoded here for development convenience.
# In a real deployment this should come from an environment variable, never committed to source.
import os
from dotenv import load_dotenv

load_dotenv()

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "fallback-dev-secret"))

# Serve CSS/JS and mount the doctor-facing HTML frontend routes
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.include_router(frontend_router)
app.include_router(patient_router)

# Load both models ONCE at startup, not per-request (much faster)
inference_engine = MedFusionInference()

UPLOAD_DIR = "backend/app/uploaded_ecgs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "MedFusion backend is running"}


@app.post("/doctors")
def create_doctor(name: str = Form(...), email: str = Form(...), password: str = Form(...),
                   db: Session = Depends(get_db)):
    doctor = Doctor(name=name, email=email, hashed_password=password)  # note: this raw API endpoint bypasses hashing — kept only for early Swagger testing, the real signup flow is /ui/login
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return {"id": doctor.id, "name": doctor.name}


@app.post("/patients")
def create_patient(name: str = Form(...), age: int = Form(...), sex: int = Form(...),
                    doctor_id: int = Form(...), db: Session = Depends(get_db)):
    patient = Patient(name=name, age=age, sex=sex, doctor_id=doctor_id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return {"id": patient.id, "name": patient.name}


@app.post("/assess")
def create_assessment(
    patient_id: int = Form(...),
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...),
    ecg_image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    image_path = os.path.join(UPLOAD_DIR, f"patient_{patient_id}_{ecg_image.filename}")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(ecg_image.file, f)

    patient_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    result = inference_engine.predict_combined(patient_dict, image_path)

    assessment = Assessment(
        patient_id=patient_id,
        trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
        exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal, cp=cp,
        ecg_image_path=image_path,
        clinical_risk=result["clinical_risk"],
        ecg_risk=result["ecg_risk"],
        final_risk_score=result["final_risk_score"],
        risk_label=result["risk_label"],
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return {**result, "assessment_id": assessment.id}


@app.get("/patients/{patient_id}/assessments")
def get_patient_assessments(patient_id: int, db: Session = Depends(get_db)):
    assessments = db.query(Assessment).filter(Assessment.patient_id == patient_id).all()
    return [
        {
            "id": a.id,
            "created_at": a.created_at,
            "final_risk_score": a.final_risk_score,
            "risk_label": a.risk_label,
        }
        for a in assessments
    ]