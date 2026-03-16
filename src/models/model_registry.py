from __future__ import annotations

from typing import Dict

from config import get_model_params
from src.utils.env import RuntimeConfig
from .base_model import BaseForecastModel
from .gru_model import GRUForecastModel
from .lstm_model import LSTMForecastModel
from .moirai_model import MoiraiZeroShotModel
from .xgboost_model import XGBoostForecastModel


def build_model_registry(input_dim: int, runtime: RuntimeConfig) -> Dict[str, BaseForecastModel]:
    lstm_cfg = get_model_params("lstm")
    gru_cfg = get_model_params("gru")
    xgb_cfg = get_model_params("xgboost")
    moirai_cfg = get_model_params("moirai")

    return {
        "lstm": LSTMForecastModel(
            input_dim=input_dim,
            hidden_dim=int(lstm_cfg.get("hidden_dim", 64)),
            num_layers=int(lstm_cfg.get("num_layers", 2)),
            learning_rate=float(lstm_cfg.get("learning_rate", 1e-3)),
        ),
        "gru": GRUForecastModel(
            input_dim=input_dim,
            hidden_dim=int(gru_cfg.get("hidden_dim", 64)),
            num_layers=int(gru_cfg.get("num_layers", 2)),
            learning_rate=float(gru_cfg.get("learning_rate", 1e-3)),
        ),
        "xgboost": XGBoostForecastModel(
            n_estimators=int(xgb_cfg.get("n_estimators", 300)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            subsample=float(xgb_cfg.get("subsample", 0.9)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
        ),
        "moirai": MoiraiZeroShotModel(
            lightweight_mode=bool(moirai_cfg.get("lightweight_mode", runtime.lightweight_mode)),
        ),
    }
