from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from config import get_path, load_settings
from .forecaster_manager import ForecasterManager


class ForecastService:
    """Facade service consumed directly by Streamlit app."""

    def __init__(self, data_path: str = "data/data.csv", model_dir: str = "models", lookback: int = 24):
        settings = load_settings()
        default_lookback = int(settings.get("data", {}).get("lookback", lookback))
        data_path = get_path("data_path", data_path) if data_path == "data/data.csv" else data_path
        model_dir = get_path("model_dir", model_dir) if model_dir == "models" else model_dir
        lookback = default_lookback if lookback == 24 else lookback

        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.lookback = lookback
        self.manager = ForecasterManager(data_path=str(self.data_path), model_dir=str(self.model_dir), lookback=lookback)

    def train(
        self,
        selected_models: List[str],
        epochs: int,
        batch_size: int,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        return self.manager.train_models(
            selected_models=selected_models,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    def predict_by_horizon(self, horizon: str) -> Dict[str, Any]:
        return self.manager.predict_horizon(horizon)
