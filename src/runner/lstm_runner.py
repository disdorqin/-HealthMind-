"""LSTM 模型执行器 - 核心模型实现"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
import uuid

from src.core.utils.training_progress import get_training_tracker


class LSTMModel(nn.Module):
    """PyTorch LSTM 模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output)


class LSTMPowerForecaster:
    """LSTM 功率预测器"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        if model_params is None:
            model_params = {}
        
        defaults = {
            'input_dim': 42,
            'hidden_dim': 16,
            'num_layers': 1,
            'dropout': 0.2,
            'sequence_length': 24,
            'learning_rate': 1e-3,
            'batch_size': 256,
            'epochs': 10
        }
        
        for key, val in defaults.items():
            if key not in model_params:
                model_params[key] = val
        
        self.model_params = model_params
        self.input_dim = model_params.get('input_dim', 42)
        self.sequence_length = model_params['sequence_length']
        self.learning_rate = model_params['learning_rate']
        self.batch_size = model_params['batch_size']
        self.epochs = model_params['epochs']
        self.is_trained = False
        
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout']
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray], 
                         sequence_length: int):
        """创建序列"""
        sequences_X = []
        sequences_y = [] if y is not None else None
        
        for i in range(len(X) - sequence_length + 1):
            sequences_X.append(X[i:i + sequence_length])
            if y is not None:
                sequences_y.append(y[i + sequence_length - 1])
        
        X_out = np.array(sequences_X)
        if y is not None:
            return X_out, np.array(sequences_y)
        return X_out, None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
            verbose: bool = True) -> 'LSTMPowerForecaster':
        """训练模型"""
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # If data is already 3D (samples, sequence_length, features), skip sequencing
        if X.ndim == 3:
            X_seq = X
            y_seq = y
            # Ensure y matches X length
            if len(y) != len(X):
                # If y is provided as raw target series (longer than X due to windowing), align it?
                # Usually pre-processing returns aligned X and y.
                # Assuming y is already aligned (samples,)
                pass
        else:
            X_seq, y_seq = self._create_sequences(X, y, self.sequence_length)
        
        n_val = int(len(X_seq) * validation_split)
        if n_val > 0:
            X_train_seq, y_train_seq = X_seq[:-n_val], y_seq[:-n_val]
            X_val_seq, y_val_seq = X_seq[-n_val:], y_seq[-n_val:]
        else:
            X_train_seq, y_train_seq = X_seq, y_seq
            X_val_seq, y_val_seq = None, None
        
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device).unsqueeze(1)
        
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size, shuffle=True
        )
        
        # 准备验证数据
        val_loss = None
        if X_val_seq is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device).unsqueeze(1)
            val_loader = DataLoader(
                TensorDataset(X_val_tensor, y_val_tensor),
                batch_size=self.batch_size, shuffle=False
            )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 生成唯一的任务 ID
        task_id = f"lstm_train_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        tracker = get_training_tracker()
        tracker.start_training(task_id, self.epochs, "LSTM")
        
        self.model.train()
        for epoch in range(self.epochs):
            # 训练阶段
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            train_loss = epoch_loss / len(train_loader)
            
            # 验证阶段
            if X_val_seq is not None:
                self.model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_epoch_loss += loss.item()
                val_loss = val_epoch_loss / len(val_loader)
                self.model.train()
            
            # 更新进度（同时记录训练损失和验证损失）
            metrics = {'train_loss': train_loss}
            if val_loss is not None:
                metrics['val_loss'] = val_loss
            
            tracker.update_progress(task_id, epoch + 1, metrics=metrics)
            
            if verbose and (epoch + 1) % 5 == 0:
                if val_loss is not None:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {train_loss:.6f}")
        
        # 标记训练完成
        final_metrics = {'train_loss': train_loss}
        if val_loss is not None:
            final_metrics['val_loss'] = val_loss
        tracker.finish_training(task_id, success=True, final_metrics=final_metrics)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")
        
        X = X.astype(np.float32)
        X_seq, _ = self._create_sequences(X, None, self.sequence_length)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
        }, model_path)
    
    def load_model(self, path: str) -> 'LSTMPowerForecaster':
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_params = checkpoint['model_params']
        self.is_trained = checkpoint['is_trained']
        self.sequence_length = checkpoint['sequence_length']
        self.input_dim = self.model_params.get('input_dim', 42)
        
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.model_params.get('hidden_dim', 16),
            num_layers=self.model_params.get('num_layers', 1),
            dropout=self.model_params.get('dropout', 0.2)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        return self