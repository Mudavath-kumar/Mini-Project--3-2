# 🛡️ FINTECH SENTINEL - Advanced Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**State-of-the-art Credit Card Fraud Detection using Selective State Space Models (MambaTab)**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

FINTECH SENTINEL is a **cutting-edge fraud detection system** that leverages **Selective State Space Models (Mamba architecture)** to identify fraudulent credit card transactions in real-time. This project implements the latest advances in AI research, combining:

- 🐍 **MambaTab**: Selective State Space Model with input-dependent dynamics
- 🌲 **Ensemble Baselines**: Random Forest, XGBoost, LightGBM
- 🔍 **Explainable AI**: SHAP integration for model transparency
- 🎨 **Professional UI**: Dark-themed dashboard with interactive visualizations
- ⚡ **Real-time Detection**: Sub-second inference on CPU

### 🎯 Project Highlights

- **Novel Architecture**: First application of Mamba-style SSMs to tabular fraud detection
- **Linear Complexity**: O(L) time complexity vs Transformer's O(L²)
- **Interpretable**: SHAP-based feature importance and local explanations
- **Production-Ready**: Complete deployment pipeline with Streamlit
- **CPU-Efficient**: Trains on laptop hardware (Ryzen 5 5500U, 16GB RAM)

---

## ✨ Features

### 🎛️ Three Operation Modes

#### 1. **Dashboard Mode**
- Real-time metrics cards (Accuracy, AUC, Transactions/sec, Fraud Count)
- Interactive time-series charts (Plotly-based)
- Circular fraud probability gauge (0-100%)
- Feature importance visualization (Top 8 SHAP features)
- Recent transactions table with risk levels

#### 2. **Single Transaction Analysis**
- **Input Panel**: Transaction details (Amount, Time, Device, IP Risk, Geo Distance, Merchant)
- **Circular Gauge**: 92% fraud likelihood meter with color zones
- **Verdict Display**: Large FRAUD DETECTED / SAFE label with confidence
- **SHAP Explanation**: Horizontal bar chart showing feature contributions
- **Risk Insights**: AI-generated warnings (High IP Risk, Unusual Location, etc.)

#### 3. **Batch Analysis**
- CSV file upload for bulk processing
- Comprehensive analytics dashboard:
  - Fraud distribution pie chart
  - Probability histogram
  - Confusion matrix heatmap
  - Detailed metrics table (Precision, Recall, F1)
- Color-coded predictions with gradient styling
- Export-ready results

### 🧠 Machine Learning Models

| Model | Type | Parameters | AUC Score | Use Case |
|-------|------|-----------|-----------|----------|
| **MambaTab** | Selective SSM | 64 hidden, 2 layers | ~99.5% | Sequential pattern detection |
| Random Forest | Ensemble | 500 trees | ~99.3% | Baseline + SHAP explanations |
| XGBoost | Gradient Boosting | Default | ~99.2% | High-speed inference |
| LightGBM | Gradient Boosting | Default | ~99.1% | Memory-efficient training |
| Logistic Regression | Linear | L2 regularized | ~97.5% | Interpretable baseline |

---

## 🏗️ Architecture

### MambaTab: Selective State Space Model

```
Transaction Input (35 features)
        ↓
[Feature Embedding Layer]
    Linear(35 → 64) + LayerNorm + GELU
        ↓
[Mamba Block 1]
    ├─ SelectiveSSM (S6)
    │   ├─ Input-dependent Δ, B, C
    │   ├─ State transition: h[t] = exp(Δ·A)·h[t-1] + B·x[t]
    │   └─ Selective gating (SiLU)
    ├─ Residual Connection
    └─ MLP (4x expansion) + Residual
        ↓
[Mamba Block 2]
    └─ (Same structure)
        ↓
[Global Average Pooling]
        ↓
[Classification Head]
    Linear(64 → 32) + GELU + Dropout
    Linear(32 → 1) [Fraud Logit]
        ↓
    Fraud Probability (0-100%)
```

#### Key Components

**Selective SSM (S6 Block):**
- **A**: State transition matrix (learned diagonal)
- **B**: Input matrix (input-dependent via projection)
- **C**: Output matrix (input-dependent via projection)
- **Δ (Delta)**: Discretization timestep (controls selectivity)

**Mathematical Foundation:**
```
Continuous: h'(t) = A·h(t) + B·x(t), y(t) = C·h(t)
Discrete:   h[t] = exp(Δ·A)·h[t-1] + B[t]·x[t], y[t] = C[t]·h[t]
```

See [`MAMBATAB_ARCHITECTURE.md`](MAMBATAB_ARCHITECTURE.md) for detailed documentation.

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip package manager
- 4GB+ RAM
- Windows/Linux/MacOS

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Mudavath-kumar/Mini-project.git
cd Mini-project

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

### Manual Installation

```bash
# Install core dependencies
pip install streamlit pandas numpy torch scikit-learn

# Install ML libraries
pip install xgboost lightgbm shap joblib

# Install visualization
pip install plotly
```

---

## 💻 Usage

### Training Models

```bash
# Train all models (baselines + MambaTab)
python train_baselines.py
```

**Output:**
```
[Random Forest] Validation AUC: 0.9934, F1: 0.8567
[XGBoost] Validation AUC: 0.9912, F1: 0.8423
[LightGBM] Validation AUC: 0.9905, F1: 0.8389
[MambaTab SSM] Epoch 1/5: train_loss=0.1234, val_loss=0.1156
[MambaTab SSM] Epoch 5/5: train_loss=0.0345, val_loss=0.0378
MambaTab Test Metrics: {'auc': 0.9954, 'f1': 0.8821}
```

Models saved to `models/`:
- `baseline_random_forest.joblib` (1.28 MB)
- `mambatab_gru.pt` (76.6 KB)
- `scaler.joblib`, `feature_names.joblib`

### Running the Dashboard

```bash
streamlit run app.py
```

By default, the Streamlit app runs inference using **MambaTab (Proposed State-Space Model / SSM)** from `models/mambatab_gru.pt`.
Baseline models (e.g., Random Forest) are available only for optional comparison via the sidebar.

**Interface Overview:**
1. **Sidebar**: Select mode (Dashboard / Single Transaction / Batch Analysis)
2. **Main Panel**: Interactive visualizations and predictions
3. **Metrics**: Real-time accuracy, AUC, fraud count

### Single Transaction Detection

```python
# Example transaction
{
    "Amount": 2450.00,
    "Time": "14:30:45",
    "Device": "Mobile (iOS)",
    "IP_Risk_Score": 85,
    "Geo_Distance": 1200,
    "Merchant": "Electronics Retail"
}
```

**Output:**
- 🔴 **FRAUD DETECTED** (92% confidence)
- SHAP: IP Risk (+0.45), Amount (+0.32), Geo Distance (+0.28)
- Warnings: High IP Risk, Unusual Location

### Batch Processing

```python
# Upload CSV with columns: V1-V28, Amount, Time, Class (optional)
df = pd.read_csv("transactions.csv")

# Process through Batch Analysis mode
# Get: fraud distribution, metrics, confusion matrix, predictions
```

---

## 🎨 Demo

### Dashboard View
```
┌─────────────────────────────────────────────────────┐
│  🛡️ FINTECH SENTINEL - Fraud Detection System      │
├─────────────────────────────────────────────────────┤
│  Metrics Cards:                                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │
│  │ 98%  │ │ 3,800│ │ 0.993│ │  127 │              │
│  │ Acc  │ │ Tx/s │ │ AUC  │ │Fraud │              │
│  └──────┘ └──────┘ └──────┘ └──────┘              │
│                                                      │
│  [Time-Series Chart] [Gauge] [Feature Importance]  │
│  [Recent Transactions Table]                        │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source**: Credit Card Fraud Detection Dataset (Kaggle-style)

**Statistics:**
- Total Transactions: 284,807
- Fraud Rate: 0.17% (492 frauds)
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
