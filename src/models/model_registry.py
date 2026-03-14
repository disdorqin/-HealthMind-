from __future__ import annotations

from typing import Dict

from src.utils.env import RuntimeConfig
from .base_model import BaseForecastModel
from .gru_model import GRUForecastModel
from .lstm_model import LSTMForecastModel
from .moirai_model import MoiraiZeroShotModel
from .xgboost_model import XGBoostForecastModel


def build_model_registry(input_dim: int, runtime: RuntimeConfig) -> Dict[str, BaseForecastModel]:
    return {
        "lstm": LSTMForecastModel(input_dim=input_dim),
        "gru": GRUForecastModel(input_dim=input_dim),
        "xgboost": XGBoostForecastModel(),
        "moirai": MoiraiZeroShotModel(lightweight_mode=runtime.lightweight_mode),
    }
