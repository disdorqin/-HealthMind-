"""多模型训练器"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from src.core.utils.logger import logger
from .lstm_model import LSTMModel, CNNModel, GRUModel


class MultiModelTrainer:
    """多模型训练器 - 支持多种模型的统一训练接口"""
    
    def __init__(self, input_dim: int, output_dim: int = 1, 
                 lookback: int = 24, device: str = 'cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback = lookback
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.histories = {}
        
        logger.info(f"多模型训练器初始化，设备：{self.device}")
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                         lookback: int = 24) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """创建序列数据"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i + lookback])
            if y is not None:
                y_seq.append(y[i + lookback])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     batch_size: int = 32, test_size: float = 0.2) -> Tuple:
        """准备训练数据"""
        # 创建序列
        X_seq, y_seq = self._create_sequences(X, y, self.lookback)
        
        # 划分训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=42
        )
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, X_test, y_test
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, 
                  hidden_dim: int = 64, num_layers: int = 2, 
                  epochs: int = 50, batch_size: int = 32, 
                  learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            X: 特征数据 (n_samples, n_features)
            y: 目标数据 (n_samples,)
            hidden_dim: 隐藏层维度
            num_layers: 层数
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
        
        Returns:
            字典包含模型和训练历史
        """
        logger.info("开始训练LSTM模型")
        
        train_loader, test_loader, X_test, y_test = self._prepare_data(X, y, batch_size)
        
        model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=self.output_dim
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # 评估
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    test_loss += loss.item()
            
            test_loss = test_loss / len(test_loader)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        self.models['lstm'] = model
        self.histories['lstm'] = {
            'train_loss': train_losses,
            'test_loss': test_losses
        }
        
        logger.info("LSTM模型训练完成")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
    
    def train_cnn(self, X: np.ndarray, y: np.ndarray, 
                 num_filters: int = 32, epochs: int = 50, 
                 batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """训练CNN模型"""
        logger.info("开始训练CNN模型")
        
        train_loader, test_loader, X_test, y_test = self._prepare_data(X, y, batch_size)
        
        model = CNNModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_filters=num_filters
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    test_loss += loss.item()
            
            test_loss = test_loss / len(test_loader)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"CNN - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        self.models['cnn'] = model
        self.histories['cnn'] = {
            'train_loss': train_losses,
            'test_loss': test_losses
        }
        
        logger.info("CNN模型训练完成")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
    
    def train_gru(self, X: np.ndarray, y: np.ndarray, 
                 hidden_dim: int = 64, num_layers: int = 2, 
                 epochs: int = 50, batch_size: int = 32, 
                 learning_rate: float = 0.001) -> Dict[str, Any]:
        """训练GRU模型"""
        logger.info("开始训练GRU模型")
        
        train_loader, test_loader, X_test, y_test = self._prepare_data(X, y, batch_size)
        
        model = GRUModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=self.output_dim
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    test_loss += loss.item()
            
            test_loss = test_loss / len(test_loader)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"GRU - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        self.models['gru'] = model
        self.histories['gru'] = {
            'train_loss': train_losses,
            'test_loss': test_losses
        }
        
        logger.info("GRU模型训练完成")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
    
    def save_all_models(self, save_dir: str = 'models') -> None:
        """保存所有训练的模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = save_path / f'{model_name}_model.pth'
            torch.save(model.state_dict(), model_file)
            logger.info(f"模型已保存：{model_file}")
    
    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """获取指定的模型"""
        return self.models.get(model_name)
