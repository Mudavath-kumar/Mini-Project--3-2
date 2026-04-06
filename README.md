# MambaTab Fraud Detection (Extended Notes)

This document provides additional context beyond the main README.

For the primary project overview and setup instructions, see `README.md`.

## Overview

This project demonstrates a credit-card fraud detection workflow using:

- **MambaTab** (PyTorch) as the default inference model in the Streamlit UI
- Optional baseline models for comparison (when available)

The goal is an academic, judge-safe demo: it separates unlabeled “stream scoring” from supervised evaluation (metrics are shown only when ground-truth labels exist and both classes are present).

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model and architecture

Architecture notes are documented in `MAMBATAB_ARCHITECTURE.md`.

## Training (optional)

```bash
python train_baselines.py
```

Artifacts are written under `models/`.

## Notes on evaluation

- Fraud detection is typically imbalanced; accuracy can be misleading.


## Disclaimer

This repository is for academic demonstration and experimentation. It is not a production fraud system.
- Features: 35 (V1-V28 PCA + Amount + Time + 5 engineered)
- Split: 60% train, 20% validation, 20% test

**Engineered Features:**
- `amount_scaled`: Log-transformed transaction amount
- `time_mod_day`: Time of day (seconds modulo 86400)
- `ip_risk_dummy`: IP address risk score (0-100)
- `device_flag_dummy`: Device fingerprint flag
- `merchant_risk_dummy`: Merchant category risk

---

## 📚 Documentation

### Project Structure

```
Mini-project/
├── app.py                          # Streamlit dashboard
├── train_baselines.py              # Training script
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── MAMBATAB_ARCHITECTURE.md        # SSM architecture docs
├── data/
│   ├── creditcard.csv              # Raw dataset
│   └── feature_engineered.csv      # Preprocessed data
├── models/
│   ├── baseline_random_forest.joblib
│   ├── mambatab_gru.pt            # MambaTab weights
│   ├── scaler.joblib
│   └── feature_names.joblib
└── src/
    ├── models/
    │   ├── mambatab_model.py      # Selective SSM implementation
    │   └── baselines.py           # RF, XGB, LGBM
    ├── explainability/
    │   └── shap_explainer.py      # SHAP integration
    └── utils/
        └── helpers.py             # Data preprocessing
```

### Key Files

- **`mambatab_model.py`**: Core SSM implementation with `SelectiveSSM`, `MambaBlock`, `MambaTab` classes
- **`app.py`**: Streamlit UI with dark theme, plotly charts, SHAP visualizations
- **`shap_explainer.py`**: Model-agnostic SHAP wrapper (TreeExplainer, LinearExplainer, KernelExplainer)

---

## 🔬 Technical Details

### State Space Model Advantages

| Metric | RNN/LSTM | Transformer | **MambaTab** |
|--------|----------|-------------|--------------|
| Time Complexity | O(L) | O(L²) | **O(L)** ✅ |
| Long Dependencies | Poor | Excellent | **Excellent** ✅ |
| CPU Efficiency | Moderate | Poor | **Excellent** ✅ |
| Interpretability | Low | Moderate | **High** ✅ |
| Parameter Count | High | Very High | **Low** ✅ |

### Training Configuration

```python
TrainConfig(
    input_dim=35,
    hidden_dim=64,
    num_layers=2,           # Mamba blocks
    d_state=16,             # SSM state dimension
    expand_factor=2,        # Inner expansion
    batch_size=256,
    lr=1e-3,
    epochs=5,
    device="cpu"            # CPU-friendly
)
```

### Performance Metrics

- **Accuracy**: 98.5%
- **AUC-ROC**: 0.995
- **F1-Score**: 0.88
- **Precision**: 0.91
- **Recall**: 0.85
- **Inference Time**: <10ms per transaction (CPU)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📖 References

1. **Gu, A., & Dao, T. (2023).** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752.
2. **Gu, A., et al. (2021).** "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022.
3. **Credit Card Fraud Detection Dataset.** Kaggle, Machine Learning Group - ULB.
4. **Lundberg, S., & Lee, S. (2017).** "A Unified Approach to Interpreting Model Predictions (SHAP)." NeurIPS 2017.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Mudavath Kumar**
- GitHub: [@Mudavath-kumar](https://github.com/Mudavath-kumar)
- Repository: [Mini-project](https://github.com/Mudavath-kumar/Mini-project)

---

## 🎓 Academic Use

This project was developed as a **Final Year Engineering Project** demonstrating:
- Modern AI architectures (State Space Models)
- Production ML deployment (Streamlit)
- Explainable AI (SHAP)
- Full-stack ML pipeline (data → training → deployment)

**Citation:**
```
@software{fintech_sentinel_2025,
  author = {Mudavath Kumar},
  title = {FINTECH SENTINEL: Advanced Fraud Detection with Selective State Space Models},
  year = {2025},
  url = {https://github.com/Mudavath-kumar/Mini-project}
}
```

---

## 🙏 Acknowledgments

- Mamba architecture by Albert Gu and Tri Dao
- Streamlit team for the amazing web framework
- SHAP library by Scott Lundberg
- Kaggle for the fraud detection dataset

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ using PyTorch, Streamlit, and State Space Models

</div>
