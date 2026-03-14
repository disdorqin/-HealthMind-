"""模型基类 - 定义统一的模型接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch

from src.core.utils.logger import logger


class BaseModel(ABC):
    """
    模型基类 - 定义统一的训练和预测接口
    
    所有模型（LSTM, XGBoost, CNN, GRU 等）都应继承此基类，
    实现统一的 train() 和 predict() 接口。
    """
    
    def __init__(self, model_name: str, model_type: str):
        """
        初始化基模型
        
        Args:
            model_name: 模型名称（如 'LSTM', 'XGBoost', 'CNN' 等）
            model_type: 模型类型分类（如 'DL', 'Tree', 'Ensemble' 等）
        """
        self.model_name = model_name
        self.model_type = model_type
        self.is_trained = False
        self.training_history = {}
        self.metrics = {}
        self.created_at = datetime.now()
        
        logger.info(f"初始化模型：{self.model_name} (类型: {self.model_type})")
    
    @abstractmethod
    def train(self, X_train: np.ndarray | torch.Tensor, 
              y_train: np.ndarray | torch.Tensor,
              X_val: Optional[np.ndarray | torch.Tensor] = None,
              y_val: Optional[np.ndarray | torch.Tensor] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features) 或序列 (n_samples, seq_len, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            **kwargs: 其他训练参数（epochs, batch_size, learning_rate 等）
        
        Returns:
            训练历史字典 {'loss': [...], 'val_loss': [...], ...}
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        进行预测
        
        Args:
            X_test: 测试特征 (n_samples, n_features) 或 (n_samples, seq_len, n_features)
        
        Returns:
            np.ndarray: 预测值 (n_samples,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径
        
        Returns:
            str: 实际保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> BaseModel:
        """
        加载模型
        
        Args:
            path: 加载路径
        
        Returns:
            BaseModel: 加载后的模型实例
        """
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能 - 基础实现
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            字典包含评估指标 (MAE, RMSE, MAPE, R²)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算 MAPE (Mean Absolute Percentage Error)
        # 避免除以零
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        self.metrics = metrics
        logger.info(f"{self.model_name} - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return metrics
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics
        }
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name={self.model_name}, "
                f"type={self.model_type}, trained={self.is_trained})")
