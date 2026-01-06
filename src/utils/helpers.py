"""Utility functions for data loading, preprocessing, and feature engineering.

This is intentionally simple but complete enough to support an end-to-end
demo and a final-year project. You can extend it further for research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplits:
	X_train: np.ndarray
	X_val: np.ndarray
	X_test: np.ndarray
	y_train: np.ndarray
	y_val: np.ndarray
	y_test: np.ndarray


def load_raw_data(path: str) -> pd.DataFrame:
	"""Load the raw credit card transaction CSV."""

	return pd.read_csv(path)


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
	"""Minimal feature engineering placeholder.

	In a full project you would compute:
	- IP risk score
	- device flag
	- geo distance
	- behaviour deviation
	- velocity features
	- merchant risk

	Here we just make sure a few numeric columns exist so the rest of the
	pipeline can run even if the original CSV only has `Time`, `Amount`,
	and `Class` like the classic Kaggle credit card dataset.
	"""

	df = df.copy()

	# Assume standard Kaggle schema if present
	if "Amount" in df.columns:
		df["amount_scaled"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-6)
	else:
		df["amount_scaled"] = 0.0

	# simple time-of-day like feature from `Time` if available
	if "Time" in df.columns:
		df["time_mod_day"] = (df["Time"] % (24 * 60 * 60)) / (24 * 60 * 60)
	else:
		df["time_mod_day"] = 0.0

	# dummy engineered signals so SHAP has a few named features
	df["ip_risk_dummy"] = np.random.rand(len(df))
	df["device_flag_dummy"] = np.random.randint(0, 2, size=len(df))
	df["merchant_risk_dummy"] = np.random.rand(len(df))

	return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = "Class") -> Tuple[np.ndarray, np.ndarray, list]:
	"""Create feature matrix X and label vector y from a raw/engineered frame.

	We take all numeric columns except the target as features.
	Returns (X, y, feature_names).
	"""

	df = basic_feature_engineering(df)

	if target_col not in df.columns:
		raise ValueError(f"Target column '{target_col}' not found in dataframe.")

	y = df[target_col].values.astype(int)
	feature_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
	X = df[feature_cols].values.astype(float)

	return X, y, feature_cols


def train_val_test_split(
	X: np.ndarray,
	y: np.ndarray,
	test_size: float = 0.2,
	val_size: float = 0.1,
	random_state: int = 42,
) -> DatasetSplits:
	"""Split into train/val/test while respecting class balance via stratification."""

	X_train, X_temp, y_train, y_temp = train_test_split(
		X,
		y,
		test_size=test_size + val_size,
		stratify=y,
		random_state=random_state,
	)

	relative_val_size = val_size / (test_size + val_size)

	X_val, X_test, y_val, y_test = train_test_split(
		X_temp,
		y_temp,
		test_size=1 - relative_val_size,
		stratify=y_temp,
		random_state=random_state,
	)

	return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def scale_splits(splits: DatasetSplits) -> Tuple[DatasetSplits, StandardScaler]:
	"""Fit a StandardScaler on train and apply to all splits."""

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(splits.X_train)
	X_val_scaled = scaler.transform(splits.X_val)
	X_test_scaled = scaler.transform(splits.X_test)

	scaled = DatasetSplits(
		X_train_scaled,
		X_val_scaled,
		X_test_scaled,
		splits.y_train,
		splits.y_val,
		splits.y_test,
	)

	return scaled, scaler


def hello() -> str:
	return "helpers working"