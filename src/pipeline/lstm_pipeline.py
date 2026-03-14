"""LSTM 预测管道 - 包含训练和预测功能"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.runner.lstm_runner import LSTMPowerForecaster


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
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "model_path": self.model_path,
            "samples_trained": self.samples_trained,
        }


class LSTMPipeline:
    """LSTM 预测管道"""
    
    def __init__(self, lookback: int = 24, hidden_dim: int = 16, 
                 num_layers: int = 1, epochs: int = 10, batch_size: int = 256,
                 target_column: str = "ROUND(A.POWER,0)"):
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[LSTMPowerForecaster] = None
        self.feature_names: list = []
        self.target_column = target_column
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建特征"""
        result = df.copy()
        
        # 时间特征
        if 'DATATIME' in result.columns:
            result['DATATIME'] = pd.to_datetime(result['DATATIME'])
            result['hour'] = result['DATATIME'].dt.hour
            result['day_of_week'] = result['DATATIME'].dt.dayofweek
            result['month'] = result['DATATIME'].dt.month
            result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        # 滞后特征
        if 'PREPOWER' in result.columns:
            for lag in [1, 3, 6, self.lookback]:
                result[f'lag_{lag}h'] = result['PREPOWER'].shift(lag)
        
        # 滚动特征
        if 'PREPOWER' in result.columns:
            result[f'rolling_mean_{self.lookback}h'] = result['PREPOWER'].rolling(window=self.lookback).mean()
            result[f'rolling_std_{self.lookback}h'] = result['PREPOWER'].rolling(window=self.lookback).std()
        
        # 删除 NaN 行
        result = result.dropna()
        
        return result
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """获取特征列名"""
        exclude_cols = ['DATATIME', 'PREPOWER']
        return [col for col in df.columns if col not in exclude_cols and col != self.target_column]
    
    def _ensure_target_exists(self, df: pd.DataFrame) -> None:
        """确保目标列存在"""
        if self.target_column not in df.columns:
            # 尝试其他可能的列名
            alternatives = ['actual_power', 'power', 'ROUND(A.POWER,0)', 'PREPOWER']
            for alt in alternatives:
                if alt in df.columns:
                    self.target_column = alt
                    return
            raise ValueError(f"Target column not found. Available: {list(df.columns)}")
    
    def train(self, data_path: str, model_save_path: str, 
              nrows: Optional[int] = None) -> LSTMPipelineResult:
        """训练模型"""
        # 加载数据
        df = pd.read_csv(data_path, nrows=nrows)
        
        # 确保目标列存在
        self._ensure_target_exists(df)
        
        # 特征工程
        processed = self._build_features(df)
        
        # 获取特征和目标
        feature_cols = self._get_feature_columns(processed)
        self.feature_names = feature_cols
        
        X = processed[feature_cols].values.astype(np.float32)
        y = processed[self.target_column].values.astype(np.float32)
        
        # 清理 NaN/Inf
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        valid_mask = valid_mask & ~np.isnan(y) & ~np.isinf(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 创建并训练模型
        self.model = LSTMPowerForecaster({
            'input_dim': X.shape[1],
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'sequence_length': self.lookback,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        })
        
        self.model.fit(X, y, validation_split=0.2, verbose=False)
        
        # 评估
        y_pred = self.model.predict(X)
        min_len = min(len(y), len(y_pred))
        
        metrics = {
            'mae': float(mean_absolute_error(y[-min_len:], y_pred[-min_len:])),
            'rmse': float(np.sqrt(mean_squared_error(y[-min_len:], y_pred[-min_len:]))),
            'r2': float(r2_score(y[-min_len:], y_pred[-min_len:])),
            'mape': float(np.mean(np.abs((y[-min_len:] - y_pred[-min_len:]) / (y[-min_len:] + 1e-8))) * 100),
        }
        
        # 保存模型
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_save_path)
        
        return LSTMPipelineResult(
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            r2=metrics['r2'],
            mape=metrics['mape'],
            model_path=model_save_path,
            samples_trained=len(X),
        )
    
    def predict(self, data_path: str, model_path: str, 
                nrows: Optional[int] = None) -> np.ndarray:
        """使用模型进行预测"""
        # 加载模型
        self.model = LSTMPowerForecaster().load_model(model_path)
        self.lookback = self.model.sequence_length
        
        # 加载并处理数据
        df = pd.read_csv(data_path, nrows=nrows)
        processed = self._build_features(df)
        
        feature_cols = self._get_feature_columns(processed)
        X = processed[feature_cols].values.astype(np.float32)
        
        # 清理 NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[valid_mask]
        
        return self.model.predict(X)