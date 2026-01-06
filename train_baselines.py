"""Train baseline fraud detection models and save artifacts for the Streamlit app.

Run this once before launching the app:

    python train_baselines.py

It will read data/creditcard.csv, build features, split & scale, train
Logistic Regression + Random Forest (+ XGBoost/LightGBM if installed),
evaluate them, and save the chosen deployment model and scaler in models/.
"""

import os
import warnings

# Prevent PyTorch from eagerly importing/initializing torch._dynamo / compile stack,
# which can be very slow or problematic on some Windows environments.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.helpers import (
    load_raw_data,
    build_feature_matrix,
    train_val_test_split,
    scale_splits,
)
from src.models.baselines import train_baselines, evaluate_model
from src.models.mambatab_model import TrainConfig, train_gru_model


# Ignore UserWarnings (e.g., from LightGBM)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("[train_baselines] Loading data from", DATA_PATH)
    df = load_raw_data(DATA_PATH)

    print("[train_baselines] Building feature matrix...")
    X, y, feature_names = build_feature_matrix(df)

    print("[train_baselines] Splitting train/val/test...")
    splits = train_val_test_split(X, y)

    print("[train_baselines] Scaling features...")
    scaled_splits, scaler = scale_splits(splits)

    print("[train_baselines] Training baseline models...")
    models = train_baselines(
        scaled_splits.X_train,
        scaled_splits.y_train,
        scaled_splits.X_val,
        scaled_splits.y_val,
    )

    # Prefer RandomForest if available, otherwise take the first model.
    model = models.get("random_forest") or next(iter(models.values()))

    print("[train_baselines] Evaluating chosen model on test set...")
    metrics = evaluate_model(model, scaled_splits.X_test, scaled_splits.y_test)
    print("[train_baselines] Test metrics:", metrics)

    # ------------------------------------------------------------------
    # Train GRU/MambaTab-like model (state-space proxy) tuned to your laptop
    # ------------------------------------------------------------------
    input_dim = scaled_splits.X_train.shape[1]
    gru_config = TrainConfig(
        input_dim=input_dim,
        hidden_dim=64,   # safe for Ryzen 5 + 16GB RAM
        num_layers=1,
        batch_size=256,
        lr=1e-3,
        epochs=5,        # increase if you want more training, 5 is a good start
        device="cpu",   # no GPU needed
    )

    print("[train_baselines] Training GRU/MambaTab-like model...")
    gru_model, gru_metrics = train_gru_model(
        scaled_splits.X_train,
        scaled_splits.y_train,
        scaled_splits.X_val,
        scaled_splits.y_val,
        gru_config,
    )
    print("[train_baselines] GRU/MambaTab validation metrics:", gru_metrics)

    # Evaluate GRU/MambaTab model on test set
    print("[train_baselines] Evaluating GRU/MambaTab model on test set...")
    # Convert scaled numpy arrays to per-row sequences for evaluation
    X_test_seq = np.expand_dims(scaled_splits.X_test, axis=1)  # (N, 1, D)
    y_test = scaled_splits.y_test

    gru_model.eval()
    import torch

    with torch.no_grad():
        logits = gru_model(torch.from_numpy(X_test_seq).float())
        proba = torch.sigmoid(logits).cpu().numpy()

    y_pred = (proba >= 0.5).astype(int)
    gru_auc = roc_auc_score(y_test, proba)
    gru_f1 = f1_score(y_test, y_pred)
    print("[train_baselines] GRU/MambaTab test metrics: AUC=%.4f, F1=%.4f" % (gru_auc, gru_f1))

    # Save artifacts used by the Streamlit app
    model_path = os.path.join(MODEL_DIR, "baseline_random_forest.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    gru_model_path = os.path.join(MODEL_DIR, "mambatab_gru.pt")

    print("[train_baselines] Saving baseline model ->", model_path)
    joblib.dump(model, model_path)
    print("[train_baselines] Saving scaler ->", scaler_path)
    joblib.dump(scaler, scaler_path)
    print("[train_baselines] Saving feature names ->", features_path)
    joblib.dump(feature_names, features_path)

    # Save GRU/MambaTab model weights (PyTorch format)
    print("[train_baselines] Saving GRU/MambaTab model ->", gru_model_path)
    torch.save(gru_model.state_dict(), gru_model_path)

    print("[train_baselines] Done.")


if __name__ == "__main__":
    main()
