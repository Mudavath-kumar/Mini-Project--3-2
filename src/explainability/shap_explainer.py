"""SHAP-based explainability helpers.

Robust defaults:
- Tree models (RF/XGB/LGBM/GBDT) -> TreeExplainer (fast, stable, no background needed)
- Linear models -> LinearExplainer (requires background)
- Others -> KernelExplainer (slow; use small background)
"""

from __future__ import annotations

from typing import Any, Tuple, Optional, Callable

import numpy as np
import shap
import numpy as np


def _is_tree_model(model: Any) -> bool:
	name = type(model).__name__.lower()
	return any(k in name for k in ["randomforest", "xgb", "lgbm", "gradientboost", "gradientboosting"])  # pragma: no cover


def _is_linear_model(model: Any) -> bool:
	name = type(model).__name__.lower()
	return any(k in name for k in ["logisticregression", "sgdclassifier", "ridge", "lasso"])  # pragma: no cover


def _ensure_2d_background(background: Optional[np.ndarray], feature_dim: int) -> np.ndarray:
	if background is None:
		# default tiny gaussian background around zero in scaled space
		return np.zeros((20, feature_dim), dtype=float)
	bg = np.asarray(background)
	if bg.ndim == 1:
		bg = bg.reshape(1, -1)
	return bg


def create_explainer(model: Any, background: Optional[np.ndarray] = None):
	"""Create a SHAP explainer appropriate for the model type.

	For tree models, background is ignored. For others it's required (we will
	synthesize a small one if not provided).
	"""

	if _is_tree_model(model):
		return shap.TreeExplainer(model)

	# Background required for linear / kernel explainers
	if background is None:
		raise ValueError("background is required for non-tree models")

	bg = _ensure_2d_background(background, feature_dim=background.shape[-1])

	if _is_linear_model(model):
		return shap.LinearExplainer(model, bg)

	# generic model-agnostic fallback; use proba[:,1] if available
	def f(X: np.ndarray) -> np.ndarray:
		if hasattr(model, "predict_proba"):
			return model.predict_proba(X)[:, 1]
		out = getattr(model, "predict", None)
		if callable(out):
			return out(X)
		raise ValueError("Model must implement predict_proba or predict")

	return shap.KernelExplainer(f, bg)


def global_importance(
	explainer, X_sample: np.ndarray, feature_names: list
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (mean_abs_shap_values, indices_sorted_desc) for global importance."""

	shap_values = explainer.shap_values(X_sample)

	# For binary classifiers TreeExplainer may return list [class0, class1]
	if isinstance(shap_values, list) and len(shap_values) == 2:
		shap_values = shap_values[1]

	mean_abs = np.abs(shap_values).mean(axis=0)
	order = np.argsort(-mean_abs)
	return mean_abs, order


def local_explanation(explainer, x_instance: np.ndarray):
	"""Return SHAP values for a single instance (1D array)."""

	x = np.asarray(x_instance)
	if x.ndim == 1:
		x = x.reshape(1, -1)
	shap_values = explainer.shap_values(x)
	if isinstance(shap_values, list) and len(shap_values) == 2:
		shap_values = shap_values[1]
	return np.asarray(shap_values)[0]


def explain() -> str:
	return "shap working"