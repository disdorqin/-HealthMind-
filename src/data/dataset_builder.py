from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SequenceDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= lookback:
        raise ValueError("Insufficient data length for the specified lookback")

    seq_X = []
    seq_y = []
    for i in range(lookback, len(X)):
        seq_X.append(X[i - lookback:i])
        seq_y.append(y[i])

    return np.asarray(seq_X, dtype=np.float32), np.asarray(seq_y, dtype=np.float32)


def split_by_time(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SequenceDatasetBundle:
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return SequenceDatasetBundle(
        X_train=X[:train_end],
        y_train=y[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
    )
