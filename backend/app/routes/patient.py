"""
Patient-facing routes for MedFusion AI.
Patients authenticate with Patient Code + Date of Birth (no separate password),
and can view only their own simplified results — never clinician-level detail.
"""

import os
from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import Patient, Assessment
from backend.app.utils.fusion import MedFusionInference
from backend.app.utils.pdf_report import generate_pdf_report
from backend.app.utils.translations import get_translator

router = APIRouter(prefix="/patient")
templates = Jinja2Templates(directory="frontend/templates")

inference_engine = MedFusionInference()

FRIENDLY_FACTOR_NAMES = {
    "Age": {"en": "your age", "hi": "आपकी उम्र", "mr": "तुमचे वय"},
    "Sex": {"en": "your biological sex", "hi": "आपका जैविक लिंग", "mr": "तुमचे जैविक लिंग"},
    "Chest Pain Type": {"en": "the type of chest pain you reported", "hi": "आपके द्वारा बताया गया सीने के दर्द का प्रकार", "mr": "तुम्ही सांगितलेला छातीत दुखण्याचा प्रकार"},
    "Resting Blood Pressure": {"en": "your resting blood pressure", "hi": "आपका विश्राम रक्तचाप", "mr": "तुमचा विश्रांतीचा रक्तदाब"},
    "Cholesterol": {"en": "your cholesterol level", "hi": "आपका कोलेस्ट्रॉल स्तर", "mr": "तुमची कोलेस्टेरॉल पातळी"},
    "Fasting Blood Sugar": {"en": "your fasting blood sugar level", "hi": "आपका उपवास रक्त शर्करा स्तर", "mr": "तुमची उपाशीपोटी रक्तशर्करा पातळी"},
    "Resting ECG": {"en": "your resting ECG pattern", "hi": "आपका विश्राम ईसीजी पैटर्न", "mr": "तुमचा विश्रांतीचा ईसीजी नमुना"},
    "Max Heart Rate": {"en": "your maximum heart rate during activity", "hi": "गतिविधि के दौरान आपकी अधिकतम हृदय गति", "mr": "क्रियाकलापादरम्यान तुमचा कमाल हृदयगती दर"},
    "Exercise-Induced Angina": {"en": "chest pain triggered by exercise", "hi": "व्यायाम से उत्पन्न सीने का दर्द", "mr": "व्यायामामुळे होणारे छातीत दुखणे"},
    "ST Depression": {"en": "a specific ECG measurement (ST depression)", "hi": "एक विशिष्ट ईसीजी माप (एसटी डिप्रेशन)", "mr": "एक विशिष्ट ईसीजी मोजमाप (एसटी डिप्रेशन)"},
    "ST Slope": {"en": "your heart's electrical activity pattern during exercise", "hi": "व्यायाम के दौरान आपके हृदय की विद्युत गतिविधि पैटर्न", "mr": "व्यायामादरम्यान तुमच्या हृदयाच्या विद्युत क्रियाकलापाचा नमुना"},
    "Major Vessels (ca)": {"en": "the number of major blood vessels showing narrowing", "hi": "संकुचन दिखाने वाली प्रमुख रक्त वाहिकाओं की संख्या", "mr": "अरुंद दिसणाऱ्या प्रमुख रक्तवाहिन्यांची संख्या"},
    "Thalassemia": {"en": "a blood-flow related ECG measurement (thalassemia)", "hi": "रक्त प्रवाह से संबंधित ईसीजी माप (थैलेसीमिया)", "mr": "रक्तप्रवाहाशी संबंधित ईसीजी मोजमाप (थॅलेसेमिया)"},
}


def get_current_patient(request: Request, db: Session):
    patient_id = request.session.get("patient_id")
    if not patient_id:
        return None
    return db.query(Patient).filter(Patient.id == patient_id).first()


@router.get("/set-language/{lang_code}")
def set_language(lang_code: str, request: Request):
    if lang_code not in ["en", "hi", "mr"]:
        lang_code = "en"
    request.session["lang"] = lang_code
    referer = request.headers.get("referer", "/patient/login")
    return RedirectResponse(url=referer, status_code=303)


@router.get("/login")
def patient_login_page(request: Request):
    lang = request.session.get("lang", "en")
    t = get_translator(lang)
    return templates.TemplateResponse(request, "patient_login.html", {"t": t, "lang": lang})


@router.post("/login")
def patient_login_submit(request: Request, patient_code: str = Form(...),
                          date_of_birth: str = Form(...), db: Session = Depends(get_db)):
    lang = request.session.get("lang", "en")
    t = get_translator(lang)

    patient = db.query(Patient).filter(
        Patient.patient_code == patient_code.strip().upper(),
        Patient.date_of_birth == date_of_birth
    ).first()

    if not patient:
        return templates.TemplateResponse(
            request, "patient_login.html",
            {"error": t["login_error"], "t": t, "lang": lang}
        )

    request.session["patient_id"] = patient.id
    return RedirectResponse(url="/patient/dashboard", status_code=303)


@router.get("/logout")
def patient_logout(request: Request):
    request.session.pop("patient_id", None)
    return RedirectResponse(url="/patient/login", status_code=303)


@router.get("/dashboard")
def patient_dashboard(request: Request, db: Session = Depends(get_db)):
    patient = get_current_patient(request, db)
    if not patient:
        return RedirectResponse(url="/patient/login", status_code=303)

    lang = request.session.get("lang", "en")
    t = get_translator(lang)

    assessments = (
        db.query(Assessment)
        .filter(Assessment.patient_id == patient.id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        request, "patient_dashboard.html",
        {"patient": patient, "assessments": assessments, "t": t, "lang": lang}
    )


@router.get("/assessment/{assessment_id}")
def patient_assessment_detail(assessment_id: int, request: Request, db: Session = Depends(get_db)):
    patient = get_current_patient(request, db)
    if not patient:
        return RedirectResponse(url="/patient/login", status_code=303)

    lang = request.session.get("lang", "en")
    t = get_translator(lang)

    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id, Assessment.patient_id == patient.id
    ).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Report not found")

    patient_dict = {
        "age": patient.age, "sex": patient.sex, "cp": assessment.cp,
        "trestbps": assessment.trestbps, "chol": assessment.chol, "fbs": assessment.fbs,
        "restecg": assessment.restecg, "thalach": assessment.thalach, "exang": assessment.exang,
        "oldpeak": assessment.oldpeak, "slope": assessment.slope, "ca": assessment.ca,
        "thal": assessment.thal,
    }
    _, top_features = inference_engine.predict_clinical(patient_dict)

    increasing = [FRIENDLY_FACTOR_NAMES.get(f["feature"], {}).get(lang, f["feature"].lower())
                  for f in top_features if f["direction"] == "increases risk"][:3]
    decreasing = [FRIENDLY_FACTOR_NAMES.get(f["feature"], {}).get(lang, f["feature"].lower())
                  for f in top_features if f["direction"] == "decreases risk"][:2]

    simple_message = t.get(f"simple_{assessment.risk_label}", "")
    tips = t.get(f"tips_{assessment.risk_label}", [])

    # Trend: compare to the previous assessment, if one exists
    previous = (
        db.query(Assessment)
        .filter(Assessment.patient_id == patient.id, Assessment.created_at < assessment.created_at)
        .order_by(Assessment.created_at.desc())
        .first()
    )

    trend_message = None
    if previous:
        prev_score = previous.final_risk_score
        curr_score = assessment.final_risk_score
        prev_date = previous.created_at.strftime("%d %b %Y")

        if curr_score < prev_score:
            trend_message = t["trend_decreased"].format(prev=prev_score, curr=curr_score, date=prev_date)
        elif curr_score > prev_score:
            trend_message = t["trend_increased"].format(prev=prev_score, curr=curr_score, date=prev_date)
        else:
            trend_message = t["trend_same"].format(date=prev_date)

    # Short plain-text summary (for SMS/WhatsApp/copy)
    short_summary = (
        f"MedFusion AI Report for {patient.name} "
        f"({assessment.created_at.strftime('%d %b %Y')}): "
        f"Risk Level {t['risk_label_' + assessment.risk_label]} "
        f"({assessment.final_risk_score}%). "
        f"{t.get('next_step_text', '') if assessment.risk_label in ['MODERATE', 'HIGH'] else ''} "
        f"Not a diagnosis."
    ).strip()

    return templates.TemplateResponse(
        request, "patient_result.html",
        {
            "patient": patient,
            "assessment": assessment,
            "simple_message": simple_message,
            "tips": tips,
            "increasing_factors": increasing,
            "decreasing_factors": decreasing,
            "trend_message": trend_message,
            "short_summary": short_summary,
            "t": t,
            "lang": lang,
        }
    )


@router.get("/assessment/{assessment_id}/report")
def patient_download_report(assessment_id: int, request: Request, db: Session = Depends(get_db)):
    patient = get_current_patient(request, db)
    if not patient:
        return RedirectResponse(url="/patient/login", status_code=303)

    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id, Assessment.patient_id == patient.id
    ).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Report not found")

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