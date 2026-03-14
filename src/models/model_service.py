from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.utils.logger import logger
from src.data import DataService
from src.utils.env import RuntimeConfig, detect_runtime_config
from .model_registry import build_model_registry
from .stacking_manager import StackingManager

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


class ModelService:
    """Direct-call model service for Streamlit frontend (no Flask required)."""

    def __init__(
        self,
        data_path: str = "data/data.csv",
        model_dir: str = "models",
        lookback: int = 24,
        runtime: Optional[RuntimeConfig] = None,
    ):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.lookback = lookback
        self.runtime = runtime or detect_runtime_config()
        self.data_service = DataService(lookback=lookback)
        self.models: Dict[str, Any] = {}
        self.stacking = StackingManager(alpha=1.0)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def train(
        self,
        selected_models: List[str],
        epochs: int = 20,
        batch_size: int = 64,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        prepared = self.data_service.prepare_datasets(self.data_path)
        bundle = prepared["bundle"]
        self.models = build_model_registry(prepared["input_dim"], self.runtime)

        selected_models = [m for m in selected_models if m in self.models]
        if not selected_models:
            raise ValueError("No valid models selected")

        result: Dict[str, Any] = {
            "runtime": {
                "is_streamlit_cloud": self.runtime.is_streamlit_cloud,
                "lightweight_mode": self.runtime.lightweight_mode,
            },
            "models": {},
        }

        val_predictions: Dict[str, np.ndarray] = {}

        total = len(selected_models)
        for idx, model_name in enumerate(selected_models, start=1):
            model = self.models[model_name]

            def _local_progress(payload: Dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                payload = dict(payload)
                payload["model_index"] = idx
                payload["model_total"] = total
                progress_callback(payload)

            train_result = model.train(
                bundle.X_train,
                bundle.y_train,
                bundle.X_val,
                bundle.y_val,
                progress_callback=_local_progress,
                epochs=epochs,
                batch_size=batch_size,
            )

            val_pred = model.predict(bundle.X_val)
            test_pred = model.predict(bundle.X_test)
            val_predictions[model_name] = val_pred

            val_metrics = self._metrics(bundle.y_val, val_pred)
            test_metrics = self._metrics(bundle.y_test, test_pred)

            model_path = self.model_dir / f"{model_name}_model.bin"
            model.save(model_path)

            result["models"][model_name] = {
                "train": train_result,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "model_path": str(model_path),
            }

        if len(val_predictions) >= 2:
            stack_metrics = self.stacking.train(val_predictions, bundle.y_val)
            stacking_path = self.model_dir / "stacking_meta.bin"
            self.stacking.save(stacking_path)
            result["stacking"] = {
                "enabled": True,
                "metrics": stack_metrics,
                "model_path": str(stacking_path),
            }
        else:
            result["stacking"] = {"enabled": False, "reason": "Need at least 2 base models"}

        result["elapsed_seconds"] = round(time.time() - t0, 3)
        logger.info("ModelService training completed")
        return result

    def predict(
        self,
        selected_models: List[str],
        use_stacking: bool = True,
        horizon: int = 96,
    ) -> Dict[str, Any]:
        prepared = self.data_service.prepare_datasets(self.data_path)
        bundle = prepared["bundle"]

        if not self.models:
            self.models = build_model_registry(prepared["input_dim"], self.runtime)
            for name, model in self.models.items():
                model_path = self.model_dir / f"{name}_model.bin"
                if model_path.exists():
                    try:
                        model.load(model_path)
                    except Exception as exc:
                        logger.warning("Failed to load model %s: %s", name, exc)

            stacking_path = self.model_dir / "stacking_meta.bin"
            if stacking_path.exists():
                try:
                    self.stacking.load(stacking_path)
                except Exception as exc:
                    logger.warning("Failed to load stacking model: %s", exc)

        selected_models = [m for m in selected_models if m in self.models]
        if not selected_models:
            raise ValueError("No valid models selected")

        per_model_preds: Dict[str, np.ndarray] = {}
        for model_name in selected_models:
            pred = self.models[model_name].predict(bundle.X_test)
            if len(pred) >= horizon:
                pred = pred[-horizon:]
            per_model_preds[model_name] = pred.astype(np.float32)

        out: Dict[str, Any] = {
            "predictions": {k: v.tolist() for k, v in per_model_preds.items()},
            "ground_truth": bundle.y_test[-horizon:].astype(np.float32).tolist(),
        }

        if use_stacking and len(per_model_preds) >= 2 and self.stacking.is_fitted:
            min_len = min(len(v) for v in per_model_preds.values())
            aligned = {k: v[-min_len:] for k, v in per_model_preds.items()}
            ensemble = self.stacking.predict(aligned)
            out["predictions"]["stacking"] = ensemble.tolist()

        return out
