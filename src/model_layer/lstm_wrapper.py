"""LSTM模型BaseModel包装 - PyTorch深度学习模型"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.core.utils.logger import logger
from .base_model import BaseModel


class LSTMModelWrapper(BaseModel):
    """
    LSTM模型包装类 - 实现BaseModel接口
    
    特点：
    - 适合时序数据预测
    - RNN结构捕获时间依赖
    - 支持序列输入
    
    示例：
        >>> model = LSTMModelWrapper(input_dim=8, hidden_dim=64, num_layers=2)
        >>> model.train(X_train, y_train, epochs=50, batch_size=32)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2, lookback: int = 24):
        """
        初始化LSTM模型
        
        Args:
            input_dim: 特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度
            dropout: Dropout比例
            lookback: 时间步长（序列长度）
        """
        super().__init__(model_name='LSTM', model_type='DL')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.lookback = lookback
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"LSTM模型初始化 - input_dim: {input_dim}, hidden_dim: {hidden_dim}, "
                    f"device: {self.device}")
    
    def _build_model(self):
        """构建LSTM神经网络"""
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0
                )
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                last_output = self.dropout(last_output)
                output = self.linear(last_output)
                return output
        
        self.model = LSTMNet(self.input_dim, self.hidden_dim, self.num_layers, 
                            self.output_dim, self.dropout).to(self.device)
        logger.info(f"LSTM网络已构建")
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建序列数据
        
        Args:
            X: 输入特征 (n_samples, n_features)
            y: 目标标签 (n_samples,) [可选]
        
        Returns:
            序列化后的X和y
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.lookback):
            X_seq.append(X[i:i + self.lookback])
            if y is not None:
                y_seq.append(y[i + self.lookback])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              verbose: bool = True,
              **kwargs) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            verbose: 是否打印训练进度
            **kwargs: 其他参数
        
        Returns:
            训练历史
        """
        logger.info(f"开始训练LSTM模型 - epochs: {epochs}, batch_size: {batch_size}")
        
        # 构建模型
        self._build_model()
        
        # 创建序列
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        logger.info(f"训练集形状: {X_train_seq.shape}")
        
        # 准备数据加载器
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证数据加载器（如果提供）
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).reshape(-1, 1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            logger.info(f"验证集形状: {X_val_seq.shape}")
        
        # 优化器和损失函数
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - "
                               f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        self.is_trained = True
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses if val_loader else None
        }
        
        logger.info(f"LSTM模型训练完成")
        return self.training_history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        进行预测
        
        Args:
            X_test: 测试特征 (n_samples, n_features)
        
        Returns:
            预测值 (n_samples,)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型未训练，请先调用 train() 方法")
        
        # 创建序列
        X_test_seq, _ = self._create_sequences(X_test)
        logger.info(f"LSTM预测。测试集形状: {X_test_seq.shape}")
        
        # 预测
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
        
        y_pred = y_pred.cpu().numpy().flatten().astype(np.float32)
        return y_pred
    
    def save(self, path: str) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径
        
        Returns:
            实际保存路径
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型未训练，无法保存")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存检查点
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'lookback': self.lookback,
            'metadata': self.get_metadata()
        }, path)
        
        logger.info(f"LSTM模型已保存到：{path}")
        return str(path)
    
    def load(self, path: str) -> LSTMModelWrapper:
        """
        加载模型
        
        Args:
            path: 加载路径
        
        Returns:
            加载后的模型实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复模型参数
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.output_dim = checkpoint['output_dim']
        self.dropout = checkpoint['dropout']
        self.lookback = checkpoint['lookback']
        
        # 重建模型
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_trained = True
        
        logger.info(f"LSTM模型已加载：{path}")
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        metadata = super().get_metadata()
        metadata.update({
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'lookback': self.lookback
        })
        return metadata
