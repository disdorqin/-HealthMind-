from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_and_engineer_features(
    csv_path: Path,
    target_col: str = "YD15",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load csv and build a robust numeric feature matrix."""
    df = pd.read_csv(csv_path)

    if "DATATIME" in df.columns:
        dt = pd.to_datetime(df["DATATIME"], errors="coerce")
        df["hour"] = dt.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["month"] = dt.dt.month.fillna(1).astype(int)

    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in input data")
        target_col = numeric_cols[-1]

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    if target_col not in numeric_df.columns:
        numeric_df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)

    y = numeric_df[target_col].to_numpy(dtype=np.float32)
    feature_df = numeric_df.drop(columns=[target_col]).copy()
    if feature_df.empty:
        feature_df["lag_target"] = pd.Series(y).shift(1).fillna(y.mean())

    X = feature_df.to_numpy(dtype=np.float32)
    return X, y, feature_df
