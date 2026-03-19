"""
HealthMind Stacking Ensemble 模型

实现心血管疾病预测的集成学习架构：
1. 基学习器：LSTM/GRU、XGBoost、Moirai(预留)
2. 元学习器：逻辑回归（场景感知融合）

性能目标：日均健康风险预测准确率 > 70%
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 配置 ==============

@dataclass
class EnsembleConfig:
    """集成学习配置"""
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # LSTM/GRU 超参数
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_epochs: int = 50
    rnn_batch_size: int = 32
    rnn_lr: float = 1e-3
    
    # XGBoost 超参数
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_lr: float = 0.1
    xgb_subsample: float = 0.8
    
    # 元学习器
    meta_learner_type: str = 'logistic'  # 'logistic' or 'ridge'
    
    # 随机种子
    random_seed: int = 42
    
    # 性能目标
    target_accuracy: float = 0.70


# ============== 基学习器 ==============

class _BidirectionalLSTM(nn.Module):
    """双层双向 LSTM 用于捕捉健康指标的周期性"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        output, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * 2, batch, hidden_dim)
        # 拼接双向最后隐藏状态
        if self.lstm.num_layers == 1:
            hidden_fwd = hidden[0, :, :]  # 前向
            hidden_bwd = hidden[1, :, :]  # 后向
        else:
            hidden_fwd = hidden[-2, :, :]  # 前向最后层
            hidden_bwd = hidden[-1, :, :]  # 后向最后层
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat).squeeze(-1)


class _BidirectionalGRU(nn.Module):
    """双层双向 GRU 用于捕捉健康指标的周期性"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        output, hidden = self.gru(x)
        # hidden: (num_layers * 2, batch, hidden_dim)
        if self.gru.num_layers == 1:
            hidden_fwd = hidden[0, :, :]  # 前向
            hidden_bwd = hidden[1, :, :]  # 后向
        else:
            hidden_fwd = hidden[-2, :, :]  # 前向最后层
            hidden_bwd = hidden[-1, :, :]  # 后向最后层
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat).squeeze(-1)


class RNNBaseLearner:
    """RNN 基学习器基类（LSTM/GRU）"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        model_type: str = 'lstm'
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'lstm':
            self.model = _BidirectionalLSTM(
                input_dim, hidden_dim, num_layers, dropout
            )
        else:
            self.model = _BidirectionalGRU(
                input_dim, hidden_dim, num_layers, dropout
            )
        self.model.to(self.device)
        self.is_fitted = False
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """训练 RNN 模型"""
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= len(train_ds)
            train_losses.append(epoch_loss)
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回概率预测"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """返回类别预测"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'model_type': self.model_type
        }, str(path))
    
    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.model_type = checkpoint['model_type']
        
        if self.model_type == 'lstm':
            self.model = _BidirectionalLSTM(
                self.input_dim, self.hidden_dim, self.num_layers
            )
        else:
            self.model = _BidirectionalGRU(
                self.input_dim, self.hidden_dim, self.num_layers
            )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.is_fitted = True


class LSTMBaseLearner(RNNBaseLearner):
    """LSTM 基学习器 - 捕捉健康指标的周期性"""
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, model_type='lstm', **kwargs)


class GRUBaseLearner(RNNBaseLearner):
    """GRU 基学习器 - 捕捉健康指标的周期性"""
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, model_type='gru', **kwargs)


class XGBoostBaseLearner:
    """
    XGBoost 基学习器 - 利用非线性交互优势
    挖掘'吸烟 + 高血压'等复合风险因素
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_importance_ = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练 XGBoost 模型"""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.feature_importance_ = self.model.feature_importances_
        self.is_fitted = True
        
        # 返回训练结果
        results = {
            'n_estimators': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        }
        return results
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回概率预测"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回类别预测"""
        return self.model.predict(X)
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))
    
    def load(self, path: Union[str, Path]) -> None:
        self.model = joblib.load(path)
        self.is_fitted = True
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        return self.feature_importance_


class MoiraiMockLearner:
    """
    Moirai 时序基础模型 Mock 接口
    用于新用户冷启动预测
    预留真实 Moirai 模型的 API 接口
    """
    
    def __init__(self, name: str = 'moirai_mock'):
        self.name = name
        self.is_fitted = False
        self.history_stats = None
        self.global_prior = 0.5
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        模拟 Moirai 零样本学习
        存储历史统计信息用于冷启动
        """
        logger.info("Initializing Moirai Mock (Zero-Shot Cold Start Mode)...")
        
        # 存储训练数据统计
        self.history_stats = {
            'mean': float(np.mean(y_train)),
            'std': float(np.std(y_train)),
            'class_ratio': float(np.mean(y_train)),
            'n_samples': len(y_train)
        }
        
        # 全局先验（用于冷启动）
        self.global_prior = self.history_stats['class_ratio']
        self.is_fitted = True
        
        return {'status': 'initialized', 'prior': self.global_prior}
    
    def predict_proba(self, X: np.ndarray, user_context: Optional[Dict] = None) -> np.ndarray:
        """
        零样本预测
        根据用户上下文动态调整预测
        """
        n_samples = len(X)
        
        if user_context is not None:
            # 场景感知：根据用户特征调整预测
            if 'risk_factors' in user_context:
                risk_boost = sum(user_context['risk_factors']) / len(user_context['risk_factors'])
                base_prob = self.global_prior * (1 + risk_boost * 0.3)
            else:
                base_prob = self.global_prior
        else:
            base_prob = self.global_prior
        
        # 添加特征相关的小幅调整
        if X.shape[1] >= 5:
            # 使用部分特征进行简单调整
            feature_signal = np.mean(X[:, :5], axis=1)
            feature_signal = (feature_signal - np.mean(feature_signal)) / (np.std(feature_signal) + 1e-6)
            probs = np.clip(base_prob + 0.1 * feature_signal, 0.01, 0.99)
        else:
            probs = np.full(n_samples, base_prob)
        
        return probs
    
    def predict(self, X: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
        """返回类别预测"""
        probs = self.predict_proba(X, **kwargs)
        return (probs >= threshold).astype(int)
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'history_stats': self.history_stats,
            'global_prior': self.global_prior,
            'name': self.name
        }, str(path))
    
    def load(self, path: Union[str, Path]) -> None:
        data = joblib.load(path)
        self.history_stats = data['history_stats']
        self.global_prior = data['global_prior']
        self.name = data.get('name', 'moirai_mock')
        self.is_fitted = True
    
    def set_api_endpoint(self, endpoint: str) -> None:
        """设置真实 Moirai API 端点（预留接口）"""
        logger.info(f"Moirai API endpoint set to: {endpoint}")
        # 实际实现时在此处添加 API 调用逻辑


# ============== 元学习器 ==============

class MetaLearner:
    """
    元学习器 - 场景感知融合
    根据输入特征动态分配基模型权重
    """
    
    def __init__(self, meta_type: str = 'logistic', **kwargs):
        self.meta_type = meta_type
        
        if meta_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif meta_type == 'ridge':
            self.model = Ridge(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown meta_type: {meta_type}")
        
        self.is_fitted = False
        self.feature_names = ['lstm', 'gru', 'xgboost', 'moirai']
        
    def train(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        训练元学习器
        
        Args:
            base_predictions: 基学习器预测结果字典
            y_true: 真实标签
            
        Returns:
            训练指标
        """
        # 构建元特征矩阵
        keys = sorted(base_predictions.keys())
        X_meta = np.column_stack([base_predictions[k].reshape(-1) for k in keys])
        self.feature_names = keys
        
        self.model.fit(X_meta, y_true)
        self.is_fitted = True
        
        # 计算训练指标
        y_pred = self.model.predict(X_meta)
        metrics = self._compute_metrics(y_true, y_pred)
        
        # 输出权重分配
        self._log_weights()
        
        return metrics
    
    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """使用元学习器进行融合预测"""
        if not self.is_fitted:
            raise RuntimeError("MetaLearner is not fitted")
        
        keys = sorted(base_predictions.keys())
        X_meta = np.column_stack([base_predictions[k].reshape(-1) for k in keys])
        
        if self.meta_type == 'logistic':
            return self.model.predict(X_meta)
        else:
            probs = self.model.predict(X_meta)
            return (probs > 0.5).astype(int)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """返回融合概率"""
        if not self.is_fitted:
            raise RuntimeError("MetaLearner is not fitted")
        
        keys = sorted(base_predictions.keys())
        X_meta = np.column_stack([base_predictions[k].reshape(-1) for k in keys])
        
        if self.meta_type == 'logistic':
            return self.model.predict_proba(X_meta)[:, 1]
        else:
            return self.model.predict(X_meta)
    
    def get_weights(self) -> Dict[str, float]:
        """获取各基模型的权重"""
        if not self.is_fitted:
            return {}
        
        if self.meta_type == 'logistic':
            weights = self.model.coef_[0]
        else:
            weights = self.model.coef_
        
        return dict(zip(self.feature_names, weights))
    
    def _log_weights(self) -> None:
        """记录权重分配"""
        weights = self.get_weights()
        logger.info("Meta-learner weights (model importance):")
        for name, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
            logger.info(f"  {name}: {weight:.4f}")
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0))
        }
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'meta_type': self.meta_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, str(path))
    
    def load(self, path: Union[str, Path]) -> None:
        data = joblib.load(path)
        self.model = data['model']
        self.meta_type = data['meta_type']
        self.feature_names = data.get('feature_names', [])
        self.is_fitted = data.get('is_fitted', True)


# ============== Stacking Ensemble ==============

class StackingEnsemble:
    """
    Stacking 集成学习架构
    
    基学习器：
    - LSTM: 双层双向 LSTM，捕捉健康指标周期性
    - GRU: 双层双向 GRU，捕捉健康指标周期性
    - XGBoost: 处理非线性交互，挖掘复合风险
    - Moirai: 时序基础模型，用于冷启动
    
    元学习器：逻辑回归，场景感知融合
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # 基学习器
        self.lstm_learner: Optional[LSTMBaseLearner] = None
        self.gru_learner: Optional[GRUBaseLearner] = None
        self.xgb_learner: Optional[XGBoostBaseLearner] = None
        self.moirai_learner: Optional[MoiraiMockLearner] = None
        
        # 元学习器
        self.meta_learner: Optional[MetaLearner] = None
        
        # 训练历史
        self.history: Dict[str, Any] = {}
        self.is_fitted = False
        
    def _init_base_learners(self, input_dim: int, seq_len: Optional[int] = None) -> None:
        """初始化基学习器"""
        # 确定输入维度
        if seq_len is not None:
            # 时序数据：(batch, seq_len, features)
            rnn_input_dim = input_dim
        else:
            # 展平数据：(batch, features)
            rnn_input_dim = input_dim
        
        self.lstm_learner = LSTMBaseLearner(
            input_dim=rnn_input_dim,
            hidden_dim=self.config.rnn_hidden_dim,
            num_layers=self.config.rnn_num_layers,
            dropout=self.config.rnn_dropout,
            learning_rate=self.config.rnn_lr
        )
        
        self.gru_learner = GRUBaseLearner(
            input_dim=rnn_input_dim,
            hidden_dim=self.config.rnn_hidden_dim,
            num_layers=self.config.rnn_num_layers,
            dropout=self.config.rnn_dropout,
            learning_rate=self.config.rnn_lr
        )
        
        self.xgb_learner = XGBoostBaseLearner(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_lr,
            subsample=self.config.xgb_subsample
        )
        
        self.moirai_learner = MoiraiMockLearner()
        
        logger.info(f"Initialized base learners with input_dim={rnn_input_dim}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        is_sequential: bool = True
    ) -> Dict[str, Any]:
        """
        训练 Stacking 集成模型
        
        Args:
            X_train: 训练特征 (batch, features) 或 (batch, seq_len, features)
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            is_sequential: X 是否为时序数据
            
        Returns:
            训练结果和评估指标
        """
        logger.info("=" * 50)
        logger.info("Training Stacking Ensemble")
        logger.info("=" * 50)
        
        # 确定输入维度
        if is_sequential:
            input_dim = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
            seq_len = X_train.shape[1] if len(X_train.shape) == 3 else None
        else:
            input_dim = X_train.shape[1]
            seq_len = None
        
        # 初始化基学习器
        self._init_base_learners(input_dim, seq_len)
        
        # 准备 RNN 输入（如果需要）
        if not is_sequential and seq_len is not None:
            X_train_rnn = X_train.reshape(-1, seq_len, input_dim)
        elif is_sequential:
            X_train_rnn = X_train
        else:
            X_train_rnn = np.expand_dims(X_train, axis=1)
        
        X_val_rnn = X_val
        if X_val is not None:
            if not is_sequential and seq_len is not None:
                X_val_rnn = X_val.reshape(-1, seq_len, input_dim)
            elif is_sequential:
                X_val_rnn = X_val
        
        # 1. 训练基学习器
        logger.info("\n[Step 1] Training Base Learners")
        
        # LSTM
        logger.info("Training LSTM...")
        lstm_history = self.lstm_learner.train(
            X_train_rnn, y_train, X_val_rnn, y_val,
            epochs=self.config.rnn_epochs,
            batch_size=self.config.rnn_batch_size
        )
        self.history['lstm'] = lstm_history
        
        # GRU
        logger.info("Training GRU...")
        gru_history = self.gru_learner.train(
            X_train_rnn, y_train, X_val_rnn, y_val,
            epochs=self.config.rnn_epochs,
            batch_size=self.config.rnn_batch_size
        )
        self.history['gru'] = gru_history
        
        # XGBoost
        logger.info("Training XGBoost...")
        xgb_history = self.xgb_learner.train(
            X_train, y_train, X_val, y_val
        )
        self.history['xgboost'] = xgb_history
        
        # Moirai
        logger.info("Training Moirai (Zero-Shot)...")
        moirai_history = self.moirai_learner.train(X_train, y_train)
        self.history['moirai'] = moirai_history
        
        # 2. 生成元学习器训练数据
        logger.info("\n[Step 2] Generating Meta-Features")
        
        meta_train = {
            'lstm': self.lstm_learner.predict_proba(X_train_rnn),
            'gru': self.gru_learner.predict_proba(X_train_rnn),
            'xgboost': self.xgb_learner.predict_proba(X_train),
            'moirai': self.moirai_learner.predict_proba(X_train)
        }
        
        meta_val = None
        meta_val_y = None
        if X_val is not None:
            meta_val = {
                'lstm': self.lstm_learner.predict_proba(X_val_rnn),
                'gru': self.gru_learner.predict_proba(X_val_rnn),
                'xgboost': self.xgb_learner.predict_proba(X_val),
                'moirai': self.moirai_learner.predict_proba(X_val)
            }
            meta_val_y = y_val
        
        # 3. 训练元学习器
        logger.info("\n[Step 3] Training Meta-Learner")
        
        self.meta_learner = MetaLearner(meta_type=self.config.meta_learner_type)
        meta_history = self.meta_learner.train(meta_train, y_train)
        self.history['meta'] = meta_history
        
        # 4. 验证集评估
        logger.info("\n[Step 4] Validation Evaluation")
        
        val_metrics = {}
        if X_val is not None:
            val_metrics = self.evaluate(X_val, y_val, is_sequential)
            self.history['val_metrics'] = val_metrics
            
            # 检查是否达到性能目标
            if val_metrics['accuracy'] >= self.config.target_accuracy:
                logger.info(f"✓ Target accuracy achieved: {val_metrics['accuracy']:.4f} >= {self.config.target_accuracy}")
            else:
                logger.warning(
                    f"✗ Target accuracy not met: {val_metrics['accuracy']:.4f} < {self.config.target_accuracy}"
                )
        
        self.is_fitted = True
        logger.info("\n" + "=" * 50)
        logger.info("Stacking Ensemble Training Complete")
        logger.info("=" * 50)
        
        return {
            'history': self.history,
            'val_metrics': val_metrics
        }
    
    def predict(self, X: np.ndarray, is_sequential: bool = False) -> np.ndarray:
        """集成预测"""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble is not fitted")
        
        # 准备输入
        if is_sequential:
            X_rnn = X
        else:
            X_rnn = np.expand_dims(X, axis=1)
        
        # 基学习器预测
        base_preds = {
            'lstm': self.lstm_learner.predict_proba(X_rnn),
            'gru': self.gru_learner.predict_proba(X_rnn),
            'xgboost': self.xgb_learner.predict_proba(X),
            'moirai': self.moirai_learner.predict_proba(X)
        }
        
        # 元学习器融合
        return self.meta_learner.predict(base_preds)
    
    def predict_proba(self, X: np.ndarray, is_sequential: bool = False) -> np.ndarray:
        """返回集成概率"""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble is not fitted")
        
        if is_sequential:
            X_rnn = X
        else:
            X_rnn = np.expand_dims(X, axis=1)
        
        base_preds = {
            'lstm': self.lstm_learner.predict_proba(X_rnn),
            'gru': self.gru_learner.predict_proba(X_rnn),
            'xgboost': self.xgb_learner.predict_proba(X),
            'moirai': self.moirai_learner.predict_proba(X)
        }
        
        return self.meta_learner.predict_proba(base_preds)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_sequential: bool = False
    ) -> Dict[str, float]:
        """评估模型性能"""
        y_pred = self.predict(X, is_sequential)
        
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0))
        }
        
        logger.info("Evaluation Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """保存完整集成模型"""
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存基学习器
        self.lstm_learner.save(save_dir / 'lstm_base.pth')
        self.gru_learner.save(save_dir / 'gru_base.pth')
        self.xgb_learner.save(save_dir / 'xgboost_base.joblib')
        self.moirai_learner.save(save_dir / 'moirai_base.joblib')
        
        # 保存元学习器
        self.meta_learner.save(save_dir / 'meta_learner.joblib')
        
        # 保存配置
        joblib.dump({
            'config': self.config,
            'is_fitted': self.is_fitted,
            'history': self.history
        }, save_dir / 'ensemble_config.joblib')
        
        logger.info(f"Saved ensemble to {save_dir}")
    
    def load(self, path: Union[str, Path]) -> None:
        """加载完整集成模型"""
        load_dir = Path(path).parent
        
        # 加载配置
        config_data = joblib.load(load_dir / 'ensemble_config.joblib')
        self.config = config_data.get('config', EnsembleConfig())
        self.is_fitted = config_data.get('is_fitted', True)
        self.history = config_data.get('history', {})
        
        # 加载基学习器
        self.lstm_learner = LSTMBaseLearner(input_dim=10)
        self.lstm_learner.load(load_dir / 'lstm_base.pth')
        
        self.gru_learner = GRUBaseLearner(input_dim=10)
        self.gru_learner.load(load_dir / 'gru_base.pth')
        
        self.xgb_learner = XGBoostBaseLearner()
        self.xgb_learner.load(load_dir / 'xgboost_base.joblib')
        
        self.moirai_learner = MoiraiMockLearner()
        self.moirai_learner.load(load_dir / 'moirai_base.joblib')
        
        # 加载元学习器
        self.meta_learner = MetaLearner()
        self.meta_learner.load(load_dir / 'meta_learner.joblib')
        
        logger.info(f"Loaded ensemble from {load_dir}")


# ============== 便捷函数 ==============

def create_healthmind_ensemble(
    input_dim: int = 17,
    target_accuracy: float = 0.70,
    **kwargs
) -> StackingEnsemble:
    """
    创建 HealthMind Stacking 集成模型
    
    Args:
        input_dim: 输入特征维度
        target_accuracy: 目标准确率
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 StackingEnsemble
    """
    config = EnsembleConfig(
        target_accuracy=target_accuracy,
        **kwargs
    )
    return StackingEnsemble(config)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int = 17,
    test_size: float = 0.2,
    val_size: float = 0.125,
    target_accuracy: float = 0.70,
    random_state: int = 42
) -> Tuple[StackingEnsemble, Dict[str, Any]]:
    """
    一键训练和评估函数
    
    Args:
        X: 特征矩阵
        y: 标签
        input_dim: 输入特征维度
        test_size: 测试集比例
        val_size: 验证集比例（占训练集）
        target_accuracy: 目标准确率
        random_state: 随机种子
        
    Returns:
        (训练好的模型，评估结果)
    """
    # 8:2 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 创建和训练模型
    ensemble = create_healthmind_ensemble(
        input_dim=input_dim,
        target_accuracy=target_accuracy,
        random_seed=random_state
    )
    
    train_result = ensemble.train(X_train, y_train, X_val, y_val)
    
    # 测试集评估
    logger.info("\n[Test Set Evaluation]")
    test_metrics = ensemble.evaluate(X_test, y_test)
    
    result = {
        'train_result': train_result,
        'test_metrics': test_metrics,
        'data_split': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
    }
    
    return ensemble, result


# ============== 主函数 ==============

if __name__ == '__main__':
    # 示例：使用模拟数据演示完整流程
    print("=" * 60)
    print("HealthMind Stacking Ensemble - 示例运行")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 17
    
    # 模拟心血管风险特征
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = np.random.uniform(30, 80, n_samples)  # 年龄
    X[:, 1] = np.random.choice([0, 1], n_samples)  # 性别
    X[:, 5] = np.random.uniform(90, 180, n_samples)  # 收缩压
    X[:, 6] = np.random.uniform(60, 120, n_samples)  # 舒张压
    
    # 生成标签（基于部分特征的简单规则）
    risk_score = 0.3 * (X[:, 0] / 80) + 0.2 * X[:, 1] + 0.3 * (X[:, 5] / 180) + 0.2 * np.random.rand(n_samples)
    y = (risk_score > 0.5).astype(int)
    
    print(f"数据形状：X={X.shape}, y={y.shape}")
    print(f"类别分布：0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # 训练和评估
    ensemble, result = train_and_evaluate(
        X, y,
        input_dim=n_features,
        test_size=0.2,
        target_accuracy=0.70
    )
    
    print("\n" + "=" * 60)
    print("运行完成!")
    print("=" * 60)
    
    # 保存模型
    ensemble.save('models/checkpoints/healthmind_ensemble.joblib')
    print("模型已保存至 models/checkpoints/healthmind_ensemble.joblib")