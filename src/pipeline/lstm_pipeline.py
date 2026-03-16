"""LSTM 预测管道 - 包含训练和预测功能"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.runner.lstm_runner import LSTMPowerForecaster
from src.data.lstm_processing import process_data_for_lstm
from src.core.utils.logger import logger
from src.core.utils.training_progress import get_training_tracker


@dataclass
class LSTMPipelineResult:
    """训练管道结果"""
    mae: float
    rmse: float
    r2: float
    mape: float
    model_path: str
    samples_trained: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'mape': self.mape,
            'model_path': self.model_path,
            'samples_trained': self.samples_trained
        }


class LSTMPipeline:
    """LSTM 预测管道"""
    
    def __init__(self, data_path: str, model_save_path: str = 'models/lstm_forecaster.pth',
                 lookback: int = 3, hidden_dim: int = 64, num_layers: int = 2,
                 epochs: int = 50, batch_size: int = 32, feature_config: Dict[str, Any] = None):
        if not data_path:
             data_path = 'data/personal_carbon_footprint_behavior.csv'
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_config = feature_config
        self.model = None
        self.scaler = None
        
    def run(self, nrows: int = None) -> LSTMPipelineResult:
        """运行完整管道"""
        try:
            # Check training progress tracker
            tracker = None
            try:
                tracker = get_training_tracker()
                if tracker:
                    tracker.start_training(self.epochs)
            except Exception:
                pass
            
            # Process Data Independently
            logger.info(f"Processing data from {self.data_path} for LSTM...")
            X, y, target_scaler = process_data_for_lstm(self.data_path, window_size=self.lookback)
            self.scaler = target_scaler
            
            logger.info(f"Data processed. X shape: {X.shape}, y shape: {y.shape}")

            # Train Model
            self.model = LSTMPowerForecaster({
                'input_dim': X.shape[2], 
                'hidden_dim': self.hidden_dim, 
                'num_layers': self.num_layers,
                'dropout': 0.2, 
                'sequence_length': self.lookback,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            })
            
            # Using custom fit method that supports 3D input
            logger.info("Starting model training...")
            self.model.fit(X, y, validation_split=0.2, verbose=True)
            
            # Evaluate
            logger.info("Evaluating model...")
            y_pred = self.model.predict(X)
            
            # Check length alignment
            min_len = min(len(y), len(y_pred))
            y_true_scaled = y[-min_len:]
            y_pred_scaled = y_pred[-min_len:]
            
            # Inverse Transform Target if Scaled
            if self.scaler:
                 y_true = self.scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
                 y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            else:
                 y_true = y_true_scaled
                 y_pred = y_pred_scaled
            
            metrics = {
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                # Handle division by zero for mape
                'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
                'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
            }
            
            logger.info(f"Training completed. Metrics: {metrics}")
            
            # Save Model
            Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self.model, 'save_model'):
                self.model.save_model(self.model_save_path)
            else:
                torch.save(self.model.model.state_dict(), self.model_save_path)
            
            # Save scaler
            scaler_path = str(Path(self.model_save_path).with_suffix('.joblib'))
            joblib.dump(self.scaler, scaler_path)
            
            if tracker:
                tracker.finish_training()
            
            return LSTMPipelineResult(
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                r2=metrics['r2'],
                mape=metrics['mape'],
                model_path=self.model_save_path,
                samples_trained=len(X)
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e
