"""
Generates a downloadable PDF risk assessment report for a patient.
Uses reportlab to build a clean, professional-looking document.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

RISK_COLORS = {
    "HIGH": colors.HexColor("#c0392b"),
    "MODERATE": colors.HexColor("#b8860b"),
    "LOW": colors.HexColor("#1e7e34"),
}


def generate_pdf_report(patient, assessment, result):
    """
    patient: Patient DB object
    assessment: Assessment DB object (already saved)
    result: the full dict returned by MedFusionInference.predict_combined()
            (includes top_clinical_factors, ecg_class_probabilities,
            gradcam_image_path, recommendation)
    """
    filename = f"MedFusion_Report_Patient{patient.id}_{assessment.id}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm,
                             leftMargin=20 * mm, rightMargin=20 * mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1a3b5d")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#1a3b5d"),
        spaceBefore=14, spaceAfter=8
    )
    normal = styles["Normal"]

    elements = []

    elements.append(Paragraph("MedFusion AI", title_style))
    elements.append(Paragraph("Heart Disease Risk Assessment Report", styles["Heading3"]))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", normal
    ))
    elements.append(Spacer(1, 16))

    # Patient info
    elements.append(Paragraph("Patient Information", heading_style))
    patient_table_data = [
        ["Patient Code", patient.patient_code or "N/A"],
        ["Name", patient.name],
        ["Age", str(patient.age)],
        ["Sex", "Male" if patient.sex == 1 else "Female"],
    ]
    patient_table = Table(patient_table_data, colWidths=[150, 300])
    patient_table.setStyle(_basic_table_style())
    elements.append(patient_table)
    elements.append(Spacer(1, 12))

    # Clinical data
    elements.append(Paragraph("Clinical Information", heading_style))
    clinical_table_data = [
        ["Resting Blood Pressure", f"{assessment.trestbps} mm Hg"],
        ["Cholesterol", f"{assessment.chol} mg/dL"],
        ["Fasting Blood Sugar > 120 mg/dL", "Yes" if assessment.fbs == 1 else "No"],
        ["Max Heart Rate Achieved", f"{assessment.thalach}"],
        ["Exercise-Induced Angina", "Yes" if assessment.exang == 1 else "No"],
        ["ST Depression (oldpeak)", f"{assessment.oldpeak}"],
        ["Chest Pain Type", f"{assessment.cp}"],
    ]
    clinical_table = Table(clinical_table_data, colWidths=[220, 230])
    clinical_table.setStyle(_basic_table_style())
    elements.append(clinical_table)
    elements.append(Spacer(1, 16))

    # AI Assessment
    elements.append(Paragraph("AI Risk Assessment", heading_style))

    risk_color = RISK_COLORS.get(assessment.risk_label, colors.black)
    risk_style = ParagraphStyle(
        "RiskStyle", parent=styles["Heading2"], textColor=risk_color
    )
    elements.append(Paragraph(
        f"Estimated Risk: {assessment.risk_label} — {assessment.final_risk_score}%", risk_style
    ))
    elements.append(Spacer(1, 8))

    breakdown_table_data = [
        ["Component", "Contribution"],
        ["Clinical Risk", f"{assessment.clinical_risk}%"],
        ["ECG Risk", f"{assessment.ecg_risk}%"],
    ]
    breakdown_table = Table(breakdown_table_data, colWidths=[220, 230])
    breakdown_table.setStyle(_header_table_style())
    elements.append(breakdown_table)
    elements.append(Spacer(1, 16))

    # SHAP — top contributing clinical factors
    elements.append(Paragraph("Top Contributing Clinical Factors (SHAP)", heading_style))
    factor_table_data = [["Factor", "Effect"]]
    for f in result.get("top_clinical_factors", []):
        factor_table_data.append([f["feature"], f["direction"].capitalize()])
    factor_table = Table(factor_table_data, colWidths=[220, 230])
    factor_table.setStyle(_header_table_style())
    elements.append(factor_table)
    elements.append(Spacer(1, 16))

    # ECG classification breakdown
    elements.append(Paragraph("ECG Classification Breakdown", heading_style))
    ecg_table_data = [["Class", "Probability"]]
    for label, prob in result.get("ecg_class_probabilities", {}).items():
        ecg_table_data.append([label.replace("_", " "), f"{prob}%"])
    ecg_table = Table(ecg_table_data, colWidths=[220, 230])
    ecg_table.setStyle(_header_table_style())
    elements.append(ecg_table)
    elements.append(Spacer(1, 16))

    # Original ECG image
    if assessment.ecg_image_path and os.path.exists(assessment.ecg_image_path):
        elements.append(Paragraph("Submitted ECG", heading_style))
        try:
            img = Image(assessment.ecg_image_path, width=160 * mm, height=90 * mm, kind="proportional")
            elements.append(img)
            elements.append(Spacer(1, 16))
        except Exception:
            pass

    # Grad-CAM heatmap
    gradcam_path = result.get("gradcam_image_path")
    if gradcam_path and os.path.exists(gradcam_path):
        elements.append(Paragraph("ECG Attention Map (Grad-CAM)", heading_style))
        elements.append(Paragraph(
            "Highlighted regions show which parts of the ECG most influenced the model's classification.",
            normal
        ))
        elements.append(Spacer(1, 6))
        try:
            gradcam_img = Image(gradcam_path, width=100 * mm, height=100 * mm, kind="proportional")
            elements.append(gradcam_img)
            elements.append(Spacer(1, 16))
        except Exception:
            pass

    # Detailed recommendation
    elements.append(Paragraph("Recommendation", heading_style))
    rec_text = result.get("recommendation", "No recommendation available.")
    elements.append(Paragraph(rec_text, normal))
    elements.append(Spacer(1, 20))

    if assessment.doctor_notes:
        elements.append(Paragraph("Doctor's Notes", heading_style))
        elements.append(Paragraph(assessment.doctor_notes, normal))
        elements.append(Spacer(1, 20))

    # Disclaimer
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey
    )
    elements.append(Paragraph(
        "Disclaimer: This report is generated by an AI-based clinical decision-support "
        "prototype (MedFusion AI) developed for academic purposes. It is not a substitute "
        "for professional medical diagnosis. All results should be reviewed and confirmed "
        "by a qualified healthcare professional.",
        disclaimer_style
    ))

    doc.build(elements)
    return filepath


def _basic_table_style():
    return TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#3b4a5a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e3e8ee")),
    ])


def _header_table_style():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3b5d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#e3e8ee")),
    ])