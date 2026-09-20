"""
SQLAlchemy ORM models — the database tables for MedFusion AI.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.app.database import Base


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    security_question = Column(String, nullable=True)
    security_answer = Column(String, nullable=True)
    specialization = Column(String, nullable=True)
    clinic_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    name = Column(String, nullable=False)
    age = Column(Integer)
    sex = Column(Integer)  # 1 = male, 0 = female (matches dataset encoding)
    date_of_birth = Column(String, nullable=True)  # stored as "YYYY-MM-DD"
    patient_code = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="patients")
    assessments = relationship("Assessment", back_populates="patient")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Clinical inputs (stored so the assessment is reproducible/auditable)
    trestbps = Column(Float)
    chol = Column(Float)
    fbs = Column(Integer)
    restecg = Column(Integer)
    thalach = Column(Float)
    exang = Column(Integer)
    oldpeak = Column(Float)
    slope = Column(Integer)
    ca = Column(Integer)
    thal = Column(Integer)
    cp = Column(Integer)

    ecg_image_path = Column(String)
    gradcam_image_path = Column(String, nullable=True)

    # Results
    clinical_risk = Column(Float)
    ecg_risk = Column(Float)
    final_risk_score = Column(Float)
    risk_label = Column(String)
    doctor_notes = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="assessments")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    action = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)  

    patient = relationship("Patient")  