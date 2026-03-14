"""Legacy runner package.

Kept for backward compatibility with old LSTM-only scripts.
"""

from src.runner.lstm_runner import LSTMPowerForecaster
from src.runner.pipeline_router import PipelineRouter, run_pipeline

__all__ = ["PipelineRouter", "run_pipeline", "LSTMPowerForecaster"]