"""XGBoost模型包装 - 实现BaseModel接口"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.core.utils.logger import logger
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost模型 - 实现BaseModel接口
    
    特点：
    - 适合处理结构化数据
    - 速度快，性能好
    - 自动特征重要性评估
    
    示例：
        >>> model = XGBoostModel(objective='reg:squarederror', n_estimators=100)
        >>> model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=10)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(self, objective: str = 'reg:squarederror', n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1, 
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        初始化XGBoost模型
        
        Args:
            objective: 目标函数（分类:'binary:logistic', 回归:'reg:squarederror'）
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率（eta）
            subsample: 行采样比例
            colsample_bytree: 列采样比例
            random_state: 随机状态
        """
        if not HAS_XGBOOST:
            raise ImportError("需要安装xgboost：pip install xgboost")
        
        super().__init__(model_name='XGBoost', model_type='Tree')
        
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = None
        self.feature_importance = None
        
        logger.info(f"XGBoost模型初始化 - objective: {objective}, n_estimators: {n_estimators}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 10,
              verbose: bool = False,
              **kwargs) -> Dict[str, Any]:
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选，用于早停）
            y_val: 验证标签（可选，用于早停）
            early_stopping_rounds: 早停轮数
            verbose: 是否输出详细信息
            **kwargs: 其他参数
        
        Returns:
            训练历史字典
        """
        logger.info("开始训练XGBoost模型")
        
        # 准备数据
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"使用验证集。训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 创建模型
        self.model = xgb.XGBRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=1 if verbose else 0,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            eval_metric='rmse'
        )
        
        # 训练
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # 获取特征重要性
        self.feature_importance = self.model.feature_importances_
        
        # 训练历史
        self.training_history = {
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators,
            'feature_importance': self.feature_importance.tolist()
        }
        
        self.is_trained = True
        logger.info(f"XGBoost模型训练完成。共{self.training_history['best_iteration']}次迭代")
        
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
        
        logger.info(f"XGBoost预测。测试集形状: {X_test.shape}")
        y_pred = self.model.predict(X_test)
        
        return y_pred.astype(np.float32)
    
    def save(self, path: str) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径
        
        Returns:
            实际保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法保存")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和特征重要性
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metadata': self.get_metadata()
        }, path)
        
        logger.info(f"XGBoost模型已保存到：{path}")
        return str(path)
    
    def load(self, path: str) -> XGBoostModel:
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
        
        data = joblib.load(path)
        self.model = data['model']
        self.feature_importance = data['feature_importance']
        self.is_trained = True
        
        logger.info(f"XGBoost模型已加载：{path}")
        return self
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[int, float]:
        """
        获取特征重要性
        
        Args:
            top_n: 返回前N个重要特征
        
        Returns:
            特征索引到重要性的映射
        """
        if self.feature_importance is None:
            raise RuntimeError("模型未训练或不支持特征重要性")
        
        # 按重要性降序排列
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        importances = {int(idx): float(self.feature_importance[idx]) 
                      for idx in indices}
        
        return importances
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        metadata = super().get_metadata()
        metadata.update({
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        })
        return metadata


class XGBoostClassifier(BaseModel):
    """
    XGBoost分类器 - 实现BaseModel接口
    
    用于分类任务的XGBoost模型包装
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42):
        """初始化XGBoost分类器"""
        if not HAS_XGBOOST:
            raise ImportError("需要安装xgboost：pip install xgboost")
        
        super().__init__(model_name='XGBoostClassifier', model_type='Tree')
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """训练分类器"""
        logger.info("开始训练XGBoost分类器")
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=0
        )
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True
        
        logger.info("XGBoost分类器训练完成")
        return {'status': 'success'}
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        return self.model.predict(X_test).astype(np.float32)
    
    def save(self, path: str) -> str:
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法保存")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'metadata': self.get_metadata()
        }, path)
        
        logger.info(f"XGBoost分类器已保存到：{path}")
        return str(path)
    
    def load(self, path: str) -> XGBoostClassifier:
        """加载模型"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.is_trained = True
        
        logger.info(f"XGBoost分类器已加载：{path}")
        return self
