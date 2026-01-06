# HuggingFace Ready MambaTab Fraud Detection App

This workspace contains a Streamlit demo for credit card fraud detection.

- Default inference model: **MambaTab (Proposed State-Space Model / SSM)** loaded from `models/mambatab_gru.pt`
- Optional comparison: baseline models (e.g., Random Forest) can be selected in the sidebar

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For full documentation, see `README_NEW.md`.
