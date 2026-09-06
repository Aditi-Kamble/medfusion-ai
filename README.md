# MedFusion AI
Multi-Modal Deep Learning System for Early Heart Disease Risk Prediction

## Structure
- data/clinical/   -> tabular patient data (Branch 1 input)
- data/ecg/        -> ECG images (Branch 2 input)
- backend/         -> FastAPI app (models, routes, inference)
- frontend/        -> HTML/CSS doctor dashboard
- notebooks/       -> exploration & model training notebooks
- reports/         -> generated PDF risk assessment reports

## Status
- [x] Clinical dataset downloaded & explored
- [ ] ECG dataset downloaded & explored
- [ ] Branch 1 (clinical NN) built
- [ ] Branch 2 (ECG CNN) built
- [ ] Fusion model
- [ ] Explainability (SHAP + Grad-CAM)
- [ ] FastAPI backend
- [ ] Frontend dashboard
- [ ] PDF report generation
- [ ] Deployment
