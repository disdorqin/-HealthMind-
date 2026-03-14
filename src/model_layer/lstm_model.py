"""LSTM模型定义"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional

from src.core.utils.logger import logger


class LSTMModel(nn.Module):
    """LSTM模型实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.linear(last_output)
        return output
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"模型已保存到：{path}")
    
    @classmethod
    def load(cls, path: str, input_dim: int, hidden_dim: int, 
             num_layers: int, output_dim: int = 1, dropout: float = 0.2) -> LSTMModel:
        """加载模型"""
        model = cls(input_dim, hidden_dim, num_layers, output_dim, dropout)
        model.load_state_dict(torch.load(path))
        logger.info(f"模型已加载：{path}")
        return model


class CNNModel(nn.Module):
    """CNN模型用于时序预测"""
    
    def __init__(self, input_dim: int, output_dim: int = 1, num_filters: int = 32):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                               kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(num_filters * 12, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: (batch_size, output_dim)
        """
        # 转换为CNN期望的格式 (batch, channels, length)
        x = x.transpose(1, 2)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"模型已保存到：{path}")


class GRUModel(nn.Module):
    """GRU模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.linear(last_output)
        return output
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"模型已保存到：{path}")
