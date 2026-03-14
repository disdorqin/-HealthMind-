from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from .dataset_builder import SequenceDatasetBundle, build_sequences, split_by_time
from .feature_engineering import load_and_engineer_features


class DataService:
    def __init__(self, lookback: int = 24, target_col: str = "YD15"):
        self.lookback = lookback
        self.target_col = target_col

    def prepare_datasets(self, data_path: str | Path) -> Dict[str, Any]:
        csv_path = Path(data_path)
        X_raw, y_raw, feature_df = load_and_engineer_features(csv_path, target_col=self.target_col)
        X_seq, y_seq = build_sequences(X_raw, y_raw, lookback=self.lookback)
        bundle: SequenceDatasetBundle = split_by_time(X_seq, y_seq)

        return {
            "bundle": bundle,
            "feature_names": feature_df.columns.tolist(),
            "input_dim": X_seq.shape[-1],
            "lookback": self.lookback,
        }

    def latest_window(self, data_path: str | Path) -> np.ndarray:
        csv_path = Path(data_path)
        X_raw, y_raw, _ = load_and_engineer_features(csv_path, target_col=self.target_col)
        if len(X_raw) < self.lookback:
            raise ValueError("Data is too short for inference window")
        window = X_raw[-self.lookback:]
        return window.astype(np.float32)
