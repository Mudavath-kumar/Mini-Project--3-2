import joblib
import numpy as np
from typing import Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:  # optional
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional
    LGBMClassifier = None


def train_baselines(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, object]:
    """Train a set of baseline classifiers and return the fitted models.

    We keep it simple but solid: LR + RF are mandatory; XGB/LGBM are trained
    only if the libraries are available in the environment.
    """
    models: Dict[str, object] = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,   # stronger default for better accuracy
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # XGBoost (optional)
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb

    # LightGBM (optional)
    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        lgbm.fit(X_train, y_train)
        models["lightgbm"] = lgbm

    # quick validation printout (can be logged instead)
    for name, model in models.items():
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        print(f"[Baseline] {name}: AUC={auc:.4f}, F1={f1:.4f}")

    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Return basic evaluation metrics for a fitted classifier."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
