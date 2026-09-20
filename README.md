# MedFusion 

**Multi-Modal Deep Learning System for Early Heart Disease Risk Prediction**

MedFusion is a clinical decision-support prototype that combines patient
clinical data and ECG imaging through two independently trained deep learning
models, fusing their outputs at inference time to estimate heart disease risk
— complete with explainable AI (SHAP + Grad-CAM), a doctor-facing web app,
and downloadable PDF risk reports.

> This is not a diagnostic tool. It is an academic prototype intended to
> demonstrate multi-modal deep learning techniques, and all outputs require
> review by a qualified healthcare professional.

---

## How it works

          Doctor
             │
   ┌─────────┴─────────┐
   ↓                   ↓

   Clinical Information ECG Image
↓ ↓
Neural Network CNN
(ClinicalNet) (ECGNet)
↓ ↓
Clinical Risk Score ECG Risk Score
└─────────┬─────────┘
↓
Weighted Fusion (inference-time)
↓
Final Risk Score + Explanation
(SHAP factors + Grad-CAM heatmap)
↓
Risk Result Page + PDF Report


**Note on fusion:** the clinical model (UCI Cleveland Heart Disease dataset,
303 patients) and the ECG model (Mendeley ECG Images dataset, 928 images)
were trained on separate, unrelated datasets, since no public dataset pairs
clinical records with matching ECG images. Fusion therefore happens at
inference time on a single real patient's concurrent data, not through joint
end-to-end training — this keeps both branches' reported metrics honest and
reproducible. See [Limitations](#limitations) for more detail.

---

## Tech stack

- **Deep Learning:** PyTorch (feed-forward NN for clinical data, CNN for ECG images)
- **Explainability:** SHAP (clinical feature attribution), Grad-CAM (ECG attention heatmaps)
- **Backend:** FastAPI, SQLAlchemy, SQLite
- **Auth:** Session-based (Starlette SessionMiddleware), passlib (pbkdf2_sha256 password hashing)
- **Frontend:** Server-rendered HTML/CSS via Jinja2 templates
- **Reports:** ReportLab (PDF generation)
- **Config:** python-dotenv

---

## Results

| Model | Metric | Result |
|---|---|---|
| Clinical (ClinicalNet) | Test Accuracy | 81.97% |
| Clinical (ClinicalNet) | Test AUC | 0.8777 |
| Clinical (ClinicalNet) | 5-Fold CV Accuracy | 80.20% ± 4.19% |
| Clinical (ClinicalNet) | 5-Fold CV AUC | 0.8797 ± 0.0428 |
| ECG (ECGNet) | Test Accuracy | 82.26% |
| ECG (ECGNet) | Test Macro-F1 | 0.7601 |

Full evaluation plots (confusion matrices, ROC curve) and the per-class ECG
breakdown are in [`reports/`](reports/).

---

## Project structure

medfusion-ai/
├── data/
│ ├── clinical/ # UCI Cleveland Heart Disease dataset (heart.csv)
│ └── ecg/ # ECG Images dataset of Cardiac Patients (4 classes)
├── backend/
│ └── app/
│ ├── models/ # Trained model weights + PyTorch architectures
│ ├── routes/ # FastAPI routes (frontend.py — doctor-facing UI)
│ ├── utils/ # Preprocessing, fusion, explainability, PDF generation
│ ├── database.py # SQLAlchemy engine/session setup
│ ├── models_db.py # DB tables (doctors, patients, assessments)
│ └── main.py # FastAPI app entry point
├── frontend/
│ ├── templates/ # Jinja2 HTML pages
│ └── static/ # CSS
├── notebooks/ # Training scripts, evaluation, cross-validation
├── reports/ # Evaluation plots, K-fold results, generated PDFs
├── requirements.txt
└── .env # Local config (not committed — see .gitignore)


---

## Setup

**1. Clone and enter the project**
```bash
git clone https://github.com/Aditi-Kamble/medfusion-ai.git
cd medfusion-ai
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:

SESSION_SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./medfusion.db


**5. Download the ECG dataset**

Download from [Mendeley Data](https://data.mendeley.com/datasets/gwbz3fsgp8/2)
and place the 4 class folders inside `data/ecg/` (see `data/ecg/README.md`
for the expected structure).

**6. Train the models** *(pre-trained weights are already included under
`backend/app/models/`, so this step is optional unless retraining)*
```bash
python notebooks/train_clinical.py
python notebooks/train_ecg.py
```

**7. Run the app**
```bash
uvicorn backend.app.main:app --reload
```

Open **http://127.0.0.1:8000/ui/login** in your browser.

---

## Usage

1. Log in (an account is created automatically on first use)
2. Add a patient
3. Click **New Assessment**, fill in clinical data, upload an ECG image
4. View the risk result — including SHAP-explained contributing factors and
   a Grad-CAM heatmap of the ECG
5. Download a full PDF report, or view the patient's assessment history

---

## Limitations

- **Dataset size:** The clinical model was trained on 303 patients and the
  ECG model on 928 images across 4 classes — both are small by deep learning
  standards. Results should be interpreted as a proof-of-concept rather than
  clinically validated performance.
- **Independently trained branches:** No public dataset pairs clinical
  records with matching ECG images from the same patients. Fusion happens at
  inference time on real concurrent patient data, not through joint
  end-to-end training.
- **No external validation set:** Both models were evaluated on held-out
  splits from their own training data, not on data from a different
  hospital, population, or ECG machine.
- **Class imbalance and weak "History of MI" recall:** The ECG classes are
  imbalanced (284 Normal, 239 MI, 233 Abnormal Heartbeat, 172 History of MI).
  Per-class evaluation shows the model performs strongly on Normal (F1
  0.826), Abnormal Heartbeat (F1 0.917), and MI (F1 0.917), but "History of
  MI" has perfect precision (1.000) with only 0.235 recall — it is correctly
  identified only about a quarter of the time it truly occurs, likely
  confused with the visually similar MI or Normal classes. This is the
  system's clearest known weak point.
- **Fusion weighting is fixed, not learned:** The final risk score combines
  both branches with a simple 50/50 weighted rule rather than a network that
  learns optimal weighting from paired data.
- **Not for clinical use:** This is an academic prototype. All outputs are
  decision-support suggestions only and require review by a qualified
  healthcare professional, as stated on every generated report.

---

## Datasets

- **Clinical:** [UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (via GitHub mirror)
- **ECG:** Khan, Ali Haider; Hussain, Muzammil (2021), *"ECG Images dataset of Cardiac Patients"*, Mendeley Data, V2, [doi:10.17632/gwbz3fsgp8.2](https://data.mendeley.com/datasets/gwbz3fsgp8/2)

---

## Author

Aditi Kamble — MCA (AI/ML), PES's Modern College of Engineering, Pune

