from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import get_horizon_settings, get_path, load_settings
from src.model_layer.model_trainer import MultiModelTrainer
from .trade_service import HorizonTradeAdvisor


@dataclass(frozen=True)
class HorizonConfig:
    key: str
    label: str
    points: int
    source_points: int
    aggregation: str
    preferred_models: List[str]
    weights: Dict[str, float]


def _load_horizon_configs() -> Dict[str, HorizonConfig]:
    defaults: Dict[str, Dict[str, Any]] = {
        "day": {
            "label": "下一天 (96x15min)",
            "points": 96,
            "source_points": 96,
            "aggregation": "none",
            "preferred_models": ["lstm", "gru", "xgboost", "moirai"],
            "weights": {"lstm": 0.40, "gru": 0.35, "xgboost": 0.15, "moirai": 0.10},
        },
        "week": {
            "label": "一周 (168x1h)",
            "points": 168,
            "source_points": 168 * 4,
            "aggregation": "hourly",
            "preferred_models": ["xgboost", "lstm", "gru", "moirai"],
            "weights": {"xgboost": 0.40, "lstm": 0.25, "gru": 0.20, "moirai": 0.15},
        },
        "month": {
            "label": "一月 (30x1d)",
            "points": 30,
            "source_points": 30 * 96,
            "aggregation": "daily",
            "preferred_models": ["moirai", "xgboost", "lstm", "gru"],
            "weights": {"moirai": 0.50, "xgboost": 0.25, "lstm": 0.15, "gru": 0.10},
        },
    }

    cfg = get_horizon_settings()
    output: Dict[str, HorizonConfig] = {}
    for key, default in defaults.items():
        current = cfg.get(key, {}) if isinstance(cfg.get(key, {}), dict) else {}
        output[key] = HorizonConfig(
            key=key,
            label=str(current.get("label", default["label"])),
            points=int(current.get("points", default["points"])),
            source_points=int(current.get("source_points", default["source_points"])),
            aggregation=str(current.get("aggregation", default["aggregation"])),
            preferred_models=list(current.get("preferred_models", default["preferred_models"])),
            weights=dict(current.get("weights", default["weights"])),
        )
    return output


HORIZON_CONFIGS: Dict[str, HorizonConfig] = _load_horizon_configs()


class ForecasterManager:
    """Unified orchestrator for LSTM/GRU/XGBoost/Moirai under three-horizon logic."""

    def __init__(self, data_path: str = "data/data.csv", model_dir: str = "models", lookback: int = 24):
        settings = load_settings()
        default_lookback = int(settings.get("data", {}).get("lookback", lookback))
        data_path = get_path("data_path", data_path) if data_path == "data/data.csv" else data_path
        model_dir = get_path("model_dir", model_dir) if model_dir == "models" else model_dir
        lookback = default_lookback if lookback == 24 else lookback

        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.lookback = lookback
        self.model_trainer = MultiModelTrainer(input_dim=10, lookback=lookback)
        self.trade_advisor = HorizonTradeAdvisor()

    @staticmethod
    def _to_np(preds: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, values in preds.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.size > 0:
                out[name] = arr
        return out

    @staticmethod
    def _aggregate(series: np.ndarray, mode: str, target_points: int) -> np.ndarray:
        if series.size == 0:
            return np.zeros(target_points, dtype=np.float32)

        if mode == "none":
            values = series
        elif mode == "hourly":
            usable = (series.size // 4) * 4
            values = series[-usable:].reshape(-1, 4).mean(axis=1) if usable >= 4 else series
        elif mode == "daily":
            usable = (series.size // 96) * 96
            values = series[-usable:].reshape(-1, 96).mean(axis=1) if usable >= 96 else series
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

        if values.size >= target_points:
            return values[-target_points:].astype(np.float32)

        pad_value = float(values[-1]) if values.size else 0.0
        padded = np.pad(values, (0, target_points - values.size), constant_values=pad_value)
        return padded.astype(np.float32)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        n = min(y_true.size, y_pred.size)
        if n == 0:
            return {"mae": 0.0, "rmse": 0.0}
        y_true = y_true[-n:]
        y_pred = y_pred[-n:]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def train_models(
        self,
        selected_models: List[str],
        epochs: int,
        batch_size: int,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """Train selected models."""
        # Placeholder - actual training requires data loading
        # This is a simplified implementation
        results = {"trained_models": [], "histories": {}}
        
        for model_name in selected_models:
            if model_name == "lstm":
                # Placeholder training call
                results["trained_models"].append("lstm")
            elif model_name == "gru":
                results["trained_models"].append("gru")
        
        return results

    def predict_horizon(self, horizon: str) -> Dict[str, Any]:
        """Generate predictions for a given horizon."""
        key = horizon.lower().strip()
        if key not in HORIZON_CONFIGS:
            raise ValueError(f"Unsupported horizon: {horizon}")

        cfg = HORIZON_CONFIGS[key]
        
        # Placeholder predictions
        predictions = {}
        for model_name in cfg.preferred_models:
            predictions[model_name] = np.zeros(cfg.points, dtype=np.float32).tolist()
        
        gt = np.zeros(cfg.points, dtype=np.float32).tolist()
        
        ensemble = np.zeros(cfg.points, dtype=np.float32)
        weight_sum = 0.0
        for name, weight in cfg.weights.items():
            if name in predictions:
                ensemble += np.array(predictions[name]) * float(weight)
                weight_sum += float(weight)

        if weight_sum > 1e-8:
            ensemble = (ensemble / weight_sum).astype(np.float32)

        model_metrics = {name: {"mae": 0.0, "rmse": 0.0} for name in predictions}
        model_metrics["ensemble"] = {"mae": 0.0, "rmse": 0.0}
        
        trade = self.trade_advisor.advise(cfg.key, ensemble)

        return {
            "horizon": cfg.key,
            "horizon_label": cfg.label,
            "points": cfg.points,
            "predictions": predictions,
            "predictions_ensemble": ensemble.tolist(),
            "ground_truth": gt,
            "metrics": model_metrics,
            "trade": trade,
        }
