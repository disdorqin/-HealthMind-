from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.utils.logger import logger
from src.utils.env import RuntimeConfig

from .model_registry import build_model_registry
from .stacking_manager import StackingManager


class ModelLoader:
    """Unified loader for checkpointed base models and stacking meta learner."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)

    @staticmethod
    def _checkpoint_filename(model_name: str) -> str:
        ext_map = {
            "lstm": ".pth",
            "gru": ".pth",
            "xgboost": ".joblib",
            "moirai": ".joblib",
        }
        ext = ext_map.get(model_name, ".bin")
        return f"{model_name}_best{ext}"

    def _checkpoint_path(self, model_name: str) -> Path:
        return self.checkpoints_dir / self._checkpoint_filename(model_name)

    def load_base_models(
        self,
        input_dim: int,
        runtime: RuntimeConfig,
        required_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        models = build_model_registry(input_dim=input_dim, runtime=runtime)
        target_models = required_models or list(models.keys())

        missing: List[str] = []
        loaded: Dict[str, Any] = {}
        for model_name in target_models:
            model = models.get(model_name)
            if model is None:
                missing.append(f"{model_name} (unknown model)")
                continue

            path = self._checkpoint_path(model_name)
            if not path.exists():
                missing.append(f"{model_name}: missing checkpoint {path}")
                continue

            model.load(path)
            loaded[model_name] = model

        if missing:
            detail = "\n".join(missing)
            raise FileNotFoundError(f"Model checkpoints are incomplete:\n{detail}")

        return loaded

    def load_stacking(self) -> StackingManager:
        path = self.checkpoints_dir / "stacking_meta_best.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Stacking checkpoint not found: {path}")

        stacking = StackingManager(alpha=1.0)
        stacking.load(path)
        logger.info("Stacking model loaded: %s", path)
        return stacking

    def load_all(
        self,
        input_dim: int,
        runtime: RuntimeConfig,
        required_models: Optional[List[str]] = None,
        require_stacking: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[StackingManager]]:
        base_models = self.load_base_models(
            input_dim=input_dim,
            runtime=runtime,
            required_models=required_models,
        )

        stacking: Optional[StackingManager] = None
        if require_stacking:
            stacking = self.load_stacking()

        return base_models, stacking
