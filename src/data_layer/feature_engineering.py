"""特征工程管道"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.utils.logger import logger


class FeatureEngineer:
    """特征工程管道 - 处理特征生成和数据转换"""
    
    def __init__(self, lookback: int = 24, test_size: float = 0.2, random_state: int = 42):
        """
        初始化特征工程师
        
        Args:
            lookback: 序列长度
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.lookback = lookback
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = None
        self.feature_names = None
        self.processed_data = None
        
        logger.info(f"特征工程师初始化：lookback={lookback}, test_size={test_size}")
    
    def add_lag_features(self, data: pd.DataFrame, columns: list, 
                        lags: int = None) -> pd.DataFrame:
        """添加滞后特征"""
        if lags is None:
            lags = self.lookback
        
        logger.info(f"添加滞后特征：列={columns}, 滞后步数={lags}")
        
        data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in range(1, lags + 1):
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # 删除NaN行
        data = data.dropna()
        
        logger.info(f"滞后特征添加完成，数据形状：{data.shape}")
        
        return data
    
    def add_rolling_features(self, data: pd.DataFrame, columns: list, 
                           windows: list = None) -> pd.DataFrame:
        """添加滚动窗口特征"""
        if windows is None:
            windows = [7, 24]
        
        logger.info(f"添加滚动窗口特征：列={columns}, 窗口={windows}")
        
        data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for window in windows:
                    data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                    data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
        
        data = data.dropna()
        
        logger.info(f"滚动窗口特征完成，数据形状：{data.shape}")
        
        return data
    
    def normalize(self, data: pd.DataFrame, method: str = 'standard', 
                 numeric_cols: list = None, fit: bool = True) -> pd.DataFrame:
        """
        数据归一化
        
        Args:
            data: 输入数据
            method: 'standard' 或 'minmax'
            numeric_cols: 需要归一化的列
            fit: 是否重新拟合scaler
        
        Returns:
            归一化后的数据
        """
        logger.info(f"数据归一化：方法={method}, fit={fit}")
        
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data = data.copy()
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"不支持的方法：{method}")
            
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            self.feature_names = numeric_cols
            logger.info(f"✓ Scaler拟合完成，处理列数：{len(numeric_cols)}")
        
        else:
            if self.scaler is None:
                logger.warning("⚠ Scaler未拟合，使用fit=True重新拟合")
                return self.normalize(data, method, numeric_cols, fit=True)
            
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
            logger.info(f"✓ 数据转换完成")
        
        self.processed_data = data
        
        return data
    
    def create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                        lookback: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建时序数据
        
        Args:
            X: 特征数据 (n_samples, n_features)
            y: 目标数据 (n_samples,)
            lookback: 序列长度
        
        Returns:
            (X_seq, y_seq) - 序列化的数据
        """
        if lookback is None:
            lookback = self.lookback
        
        logger.info(f"创建时序数据：lookback={lookback}")
        
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i + lookback])
            if y is not None:
                y_seq.append(y[i + lookback])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        logger.info(f"✓ 序列创建完成：X_seq.shape={X_seq.shape}")
        
        return X_seq, y_seq
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = None) -> Dict[str, np.ndarray]:
        """
        划分训练和测试集
        
        Args:
            X: 特征数据
            y: 目标数据
            test_size: 测试集比例
        
        Returns:
            {'X_train': ..., 'X_test': ..., 'y_train': ..., 'y_test': ...}
        """
        if test_size is None:
            test_size = self.test_size
        
        logger.info(f"划分数据集：test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        logger.info(f"✓ 数据集划分完成：")
        logger.info(f"  - X_train: {X_train.shape}")
        logger.info(f"  - X_test: {X_test.shape}")
        logger.info(f"  - y_train: {y_train.shape}")
        logger.info(f"  - y_test: {y_test.shape}")
        
        return result
    
    def save_scaler(self, path: str = 'models/scaler.pkl'):
        """保存StandardScaler"""
        if self.scaler is None:
            logger.warning("⚠ Scaler未初始化")
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, path)
        logger.info(f"✓ Scaler已保存到：{path}")
    
    @classmethod
    def load_scaler(cls, path: str = 'models/scaler.pkl'):
        """加载StandardScaler"""
        if not Path(path).exists():
            logger.error(f"❌ Scaler文件不存在：{path}")
            return None
        
        scaler = joblib.load(path)
        logger.info(f"✓ Scaler已加载：{path}")
        
        return scaler
    
    def export_training_data(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            output_dir: str = 'data/processed',
                            batch_size: int = 32) -> Dict[str, Any]:
        """
        导出训练数据为标准格式
        
        Args:
            X_train, y_train, X_test, y_test: 训练数据
            output_dir: 输出目录
            batch_size: 批大小
        
        Returns:
            包含数据加载器和元数据的字典
        """
        logger.info(f"导出训练数据到：{output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存numpy数据
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_test.npy', y_test)
        logger.info("✓ Numpy数据已保存")
        
        # 保存Scaler
        self.save_scaler(output_path / 'scaler.pkl')
        
        # 创建PyTorch DataLoader
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 保存元数据
        metadata = {
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_test_shape': X_test.shape,
            'y_test_shape': y_test.shape,
            'batch_size': batch_size,
            'lookback': self.lookback,
            'scaler_path': str(output_path / 'scaler.pkl'),
            'feature_names': self.feature_names
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ 元数据已保存")
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'scaler': self.scaler
        }
