# MambaTab Fraud Detection (Streamlit)

This repository contains a Streamlit application that demonstrates a tabular fraud-detection pipeline.

- Default inference model: **MambaTab (PyTorch checkpoint)** loaded from `models/mambatab_gru.pt`
- Optional comparison: baseline model(s) (e.g., Random Forest) can be selected in the sidebar

Important note: credit-card fraud datasets are highly imbalanced. Accuracy can look high even with weak fraud recall, so the UI emphasizes AUC-ROC/Recall where labels are available.

## Quick start

### 1) Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

## How to use

### Modes

- **Dashboard**: live-style summary metrics and recent predictions.
- **Single Transaction**: enter one transaction and get:
	- predicted fraud probability (model output)
	- a demo-friendly risk score and decision (controlled by the threshold slider)
	- a short, human-readable “Why SAFE / Why FRAUD” explanation
- **Batch Analysis**: upload a CSV and get:
	- unlabeled stream summary (risk distribution, top high-risk rows)
	- supervised metrics (AUC/precision/recall/F1/accuracy + confusion matrix) only when labels exist and both classes are present

### Batch CSV format

- The app expects model features consistent with the saved artifacts in `models/`.
- If your CSV includes ground-truth labels, use a column name like `Class`, `class`, `label`, or `target`.
- If labels are missing (or only one class is present), the app will show risk summaries and will hide supervised metrics automatically.

## Models and artifacts

The app loads the following files from `models/`:

- `mambatab_gru.pt`: MambaTab PyTorch checkpoint (used by default)
- `scaler.joblib`: preprocessing scaler used during training
- `feature_names.joblib`: the exact feature ordering expected by the model
- `baseline_random_forest.joblib`: optional baseline for comparison

## Training (optional)

If you want to retrain baselines / regenerate artifacts:

```bash
python train_baselines.py
```

Outputs are written under `models/`.

## Project structure

- `app.py`: Streamlit UI and end-to-end inference
- `src/models/`: model definitions (including MambaTab)
- `src/training/`: training utilities
- `src/explainability/`: explainability helpers
- `data/`: sample datasets

## Documentation

- Architecture notes: `MAMBATAB_ARCHITECTURE.md`
- Additional README (more detailed narrative): `README_NEW.md`

## Disclaimer

This project is intended for academic demonstration and research. Do not use as-is for real financial decisions without proper validation, monitoring, security, and compliance review.
