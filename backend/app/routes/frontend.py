"""
Frontend (HTML) routes for MedFusion AI — the doctor-facing UI.
"""

import os
import csv
import io
import shutil
from fastapi import APIRouter, Request, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from backend.app.database import get_db
from backend.app.models_db import Doctor, Patient, Assessment
from backend.app.utils.fusion import MedFusionInference
from backend.app.utils.pdf_report import generate_pdf_report
from backend.app.models_db import AuditLog

router = APIRouter(prefix="/ui")
templates = Jinja2Templates(directory="frontend/templates")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

inference_engine = MedFusionInference()
UPLOAD_DIR = "backend/app/uploaded_ecgs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_current_doctor(request: Request, db: Session):
    doctor_id = request.session.get("doctor_id")
    if not doctor_id:
        return None
    return db.query(Doctor).filter(Doctor.id == doctor_id).first()


@router.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@router.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {})


@router.post("/register")
def register_submit(request: Request, name: str = Form(...), email: str = Form(...),
                     password: str = Form(...), security_question: str = Form(...),
                     security_answer: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(Doctor).filter(Doctor.email == email).first()
    if existing:
        return templates.TemplateResponse(
            request, "register.html",
            {"error": "An account with this email already exists. Please log in instead."}
        )

    hashed = pwd_context.hash(password)
    doctor = Doctor(
        name=name, email=email, hashed_password=hashed,
        security_question=security_question,
        security_answer=security_answer.strip().lower(),
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    request.session["doctor_id"] = doctor.id
    return RedirectResponse(url="/ui/dashboard", status_code=303)


@router.post("/login")
def login_submit(request: Request, email: str = Form(...), password: str = Form(...),
                  db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == email).first()

    if not doctor:
        return templates.TemplateResponse(
            request, "login.html", {"error": "No account found with that email. Please register first."}
        )

    if not pwd_context.verify(password, doctor.hashed_password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Incorrect password. Try again."}
        )

    request.session["doctor_id"] = doctor.id
    return RedirectResponse(url="/ui/dashboard", status_code=303)

@router.get("/forgot-password")
def forgot_password_page(request: Request, email: str = None, db: Session = Depends(get_db)):
    if not email:
        return templates.TemplateResponse(request, "forgot_password.html", {"step": "email"})

    doctor = db.query(Doctor).filter(Doctor.email == email).first()
    if not doctor or not doctor.security_question:
        return templates.TemplateResponse(
            request, "forgot_password.html",
            {"step": "email", "error": "No account with a security question found for that email."}
        )

    return templates.TemplateResponse(
        request, "forgot_password.html",
        {"step": "answer", "email": email, "question": doctor.security_question}
    )


@router.post("/forgot-password")
def forgot_password_submit(request: Request, email: str = Form(...),
                            security_answer: str = Form(...), new_password: str = Form(...),
                            db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == email).first()

    if not doctor or (doctor.security_answer or "") != security_answer.strip().lower():
        return templates.TemplateResponse(
            request, "forgot_password.html",
            {"step": "answer", "email": email, "question": doctor.security_question if doctor else "",
             "error": "That answer doesn't match. Try again."}
        )

    doctor.hashed_password = pwd_context.hash(new_password)
    db.commit()

    return templates.TemplateResponse(request, "login.html", {"error": "Password reset successful. Please log in with your new password."})

@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/ui/login", status_code=303)


@router.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patients = db.query(Patient).filter(Patient.doctor_id == doctor.id).all()

    patient_rows = []
    risk_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "None": 0}

    for p in patients:
        latest = (
            db.query(Assessment)
            .filter(Assessment.patient_id == p.id)
            .order_by(Assessment.created_at.desc())
            .first()
        )
        latest_risk = latest.risk_label if latest else None
        risk_counts[latest_risk or "None"] += 1
        patient_rows.append({"patient": p, "latest_risk": latest_risk})

    analytics = {
        "total": len(patients),
        "high": risk_counts["HIGH"],
        "moderate": risk_counts["MODERATE"],
        "low": risk_counts["LOW"],
        "unassessed": risk_counts["None"],
    }

    return templates.TemplateResponse(
        request, "dashboard.html",
        {"doctor": doctor, "patient_rows": patient_rows, "analytics": analytics}
    )


@router.get("/profile")
def doctor_profile_page(request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)
    return templates.TemplateResponse(request, "doctor_profile.html", {"doctor": doctor})


@router.post("/profile")
def doctor_profile_submit(request: Request, name: str = Form(...),
                           specialization: str = Form(""), clinic_name: str = Form(""),
                           db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    doctor.name = name
    doctor.specialization = specialization.strip() or None
    doctor.clinic_name = clinic_name.strip() or None
    db.commit()

    return templates.TemplateResponse(request, "doctor_profile.html", {"doctor": doctor, "saved": True})


@router.get("/export/patients.csv")
def export_patients_csv(request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patients = db.query(Patient).filter(Patient.doctor_id == doctor.id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Patient Code", "Patient Name", "Age", "Sex", "Date of Birth",
        "Assessment Date", "Risk Label", "Final Risk Score (%)",
        "Clinical Risk (%)", "ECG Risk (%)",
        "Resting BP", "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
        "Max Heart Rate", "Exercise Angina", "ST Depression", "ST Slope",
        "Major Vessels", "Thalassemia", "Chest Pain Type", "Doctor Notes",
    ])

    for patient in patients:
        assessments = (
            db.query(Assessment)
            .filter(Assessment.patient_id == patient.id)
            .order_by(Assessment.created_at.asc())
            .all()
        )

        if not assessments:
            writer.writerow([
                patient.patient_code, patient.name, patient.age,
                "Male" if patient.sex == 1 else "Female", patient.date_of_birth,
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ])
            continue

        for a in assessments:
            writer.writerow([
                patient.patient_code, patient.name, patient.age,
                "Male" if patient.sex == 1 else "Female", patient.date_of_birth,
                a.created_at.strftime("%Y-%m-%d %H:%M"),
                a.risk_label, a.final_risk_score, a.clinical_risk, a.ecg_risk,
                a.trestbps, a.chol, a.fbs, a.restecg, a.thalach,
                a.exang, a.oldpeak, a.slope, a.ca, a.thal, a.cp,
                a.doctor_notes or "",
            ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=medfusion_patients_{doctor.id}.csv"}
    )


@router.get("/patients/new")
def new_patient_page(request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)
    return templates.TemplateResponse(
        request, "new_patient.html", {"doctor_id": doctor.id}
    )

def log_action(db: Session, doctor_id: int, action: str, patient_id: int = None):
    entry = AuditLog(doctor_id=doctor_id, patient_id=patient_id, action=action)
    db.add(entry)
    db.commit()


@router.post("/patients/new")
def new_patient_submit(request: Request, name: str = Form(...), age: int = Form(...),
                        sex: int = Form(...), date_of_birth: str = Form(...),
                        db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = Patient(name=name, age=age, sex=sex, date_of_birth=date_of_birth, doctor_id=doctor.id)
    db.add(patient)
    db.commit()
    db.refresh(patient)

    patient.patient_code = f"MF-{1000 + patient.id}"
    db.commit()

    return RedirectResponse(url="/ui/dashboard", status_code=303)


@router.get("/patients/{patient_id}/edit")
def edit_patient_page(patient_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return templates.TemplateResponse(request, "edit_patient.html", {"patient": patient})


@router.post("/patients/{patient_id}/edit")
def edit_patient_submit(patient_id: int, request: Request, name: str = Form(...),
                         age: int = Form(...), sex: int = Form(...),
                         date_of_birth: str = Form(...), db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.name = name
    patient.age = age
    patient.sex = sex
    patient.date_of_birth = date_of_birth
    db.commit()

    return RedirectResponse(url="/ui/dashboard", status_code=303)


@router.post("/patients/{patient_id}/delete")
def delete_patient(patient_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db.query(Assessment).filter(Assessment.patient_id == patient_id).delete()
    db.delete(patient)
    db.commit()

    return RedirectResponse(url="/ui/dashboard", status_code=303)


@router.get("/assess/new")
def new_assessment_page(patient_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return templates.TemplateResponse(
        request, "new_assessment.html", {"patient": patient}
    )


@router.post("/assess/new")
def new_assessment_submit(
    request: Request,
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
    doctor_notes: str = Form(""),
    db: Session = Depends(get_db),
):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    validation_errors = []
    if not (1 <= age <= 120):
        validation_errors.append("Age must be between 1 and 120.")
    if not (0 <= cp <= 3):
        validation_errors.append("Chest Pain Type must be between 0 and 3.")
    if not (50 <= trestbps <= 250):
        validation_errors.append("Resting Blood Pressure should be between 50 and 250 mm Hg.")
    if not (50 <= chol <= 700):
        validation_errors.append("Cholesterol should be between 50 and 700 mg/dL.")
    if not (0 <= restecg <= 2):
        validation_errors.append("Resting ECG Result must be between 0 and 2.")
    if not (40 <= thalach <= 250):
        validation_errors.append("Max Heart Rate should be between 40 and 250.")
    if not (0 <= oldpeak <= 10):
        validation_errors.append("ST Depression should be between 0 and 10.")
    if not (0 <= slope <= 2):
        validation_errors.append("Slope must be between 0 and 2.")
    if not (0 <= ca <= 4):
        validation_errors.append("Number of Major Vessels must be between 0 and 4.")
    if not (0 <= thal <= 3):
        validation_errors.append("Thalassemia must be between 0 and 3.")

    if validation_errors:
        return templates.TemplateResponse(
            request, "new_assessment.html",
            {"patient": patient, "errors": validation_errors}
        )

    log_action(db, doctor.id, "Ran new assessment", patient_id)

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
        gradcam_image_path=result.get("gradcam_image_path"),
        clinical_risk=result["clinical_risk"],
        ecg_risk=result["ecg_risk"],
        final_risk_score=result["final_risk_score"],
        risk_label=result["risk_label"],
        doctor_notes=doctor_notes.strip() if doctor_notes else None,
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return templates.TemplateResponse(
        request, "result.html",
        {"patient": patient, "result": result, "assessment_id": assessment.id}
    )


@router.get("/patients/{patient_id}/history")
def patient_history(patient_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    log_action(db, doctor.id, "Viewed patient history", patient_id)

    assessments = (
        db.query(Assessment)
        .filter(Assessment.patient_id == patient_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )

    chart_labels = [a.created_at.strftime("%d %b") for a in reversed(assessments)]
    chart_scores = [a.final_risk_score for a in reversed(assessments)]

    return templates.TemplateResponse(
        request, "history.html",
        {
            "patient": patient, "assessments": assessments,
            "chart_labels": chart_labels, "chart_scores": chart_scores,
        }
    )


@router.get("/assess/{assessment_id}/report")
def download_report(assessment_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    patient = db.query(Patient).filter(Patient.id == assessment.patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Not found")

    log_action(db, doctor.id, "Downloaded PDF report", assessment.patient_id)

    patient_dict = {
        "age": patient.age, "sex": patient.sex, "cp": assessment.cp,
        "trestbps": assessment.trestbps, "chol": assessment.chol, "fbs": assessment.fbs,
        "restecg": assessment.restecg, "thalach": assessment.thalach, "exang": assessment.exang,
        "oldpeak": assessment.oldpeak, "slope": assessment.slope, "ca": assessment.ca,
        "thal": assessment.thal,
    }

    result = inference_engine.predict_combined(patient_dict, assessment.ecg_image_path)

    filepath = generate_pdf_report(patient, assessment, result)

    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=os.path.basename(filepath)
    )


@router.get("/gradcam-image/{assessment_id}")
def get_gradcam_image(assessment_id: int, request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if assessment and assessment.gradcam_image_path and os.path.exists(assessment.gradcam_image_path):
        return FileResponse(assessment.gradcam_image_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Grad-CAM image not found")

@router.get("/audit-log")
def audit_log_page(request: Request, db: Session = Depends(get_db)):
    doctor = get_current_doctor(request, db)
    if not doctor:
        return RedirectResponse(url="/ui/login", status_code=303)

    logs = (
        db.query(AuditLog)
        .filter(AuditLog.doctor_id == doctor.id)
        .order_by(AuditLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return templates.TemplateResponse(request, "audit_log.html", {"logs": logs})    