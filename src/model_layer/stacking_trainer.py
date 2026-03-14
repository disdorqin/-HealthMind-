"""Stacking集成学习 - 自动多模型融合"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import joblib
from datetime import datetime

from src.core.utils.logger import logger
from .base_model import BaseModel


class StackingTrainer:
    """
    Stacking集成学习器 - 自动多模型融合
    
    工作原理：
    1. **第一层（Base Learners）**：训练多个基模型
       - LSTM、XGBoost等多种模型
       - 在训练集上学习
    
    2. **第二层（Meta Features）**：生成元特征
       - 基模型在验证集上的预测结果
       - 作为元学习器的输入
    
    3. **第三层（Meta Learner）**：训练元学习器
       - 通常是简单模型（线性回归、XGBoost）
       - 学习如何最优融合基模型预测
    
    优点：
    - 充分利用多种模型的优势
    - 显著提升预测精度
    - 减少过拟合风险
    
    示例：
        >>> stacking = StackingTrainer(
        ...     base_models=[lstm_model, xgboost_model],
        ...     meta_learner='linear'
        ... )
        >>> result = stacking.fit(X_train, y_train, X_val, y_val)
        >>> y_pred = stacking.predict(X_test)
    """
    
    def __init__(self, base_models: List[BaseModel], meta_learner: str = 'linear',
                 model_dir: str = 'models'):
        """
        初始化Stacking训练器
        
        Args:
            base_models: 基模型列表（都应实现BaseModel接口）
            meta_learner: 元学习器类型 ('linear', 'xgboost', 'ridge')
            model_dir: 模型保存目录
        """
        if not base_models:
            raise ValueError("至少需要一个基模型")
        
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_fitted = False
        self.training_history = {}
        self.final_metrics = {}
        
        logger.info(f"Stacking训练器初始化 - 基模型数: {len(base_models)}, "
                   f"元学习器: {meta_learner}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            X_test: np.ndarray = None, y_test: np.ndarray = None,
            train_kwargs: Dict[str, Any] = None,
            **kwargs) -> Dict[str, Any]:
        """
        训练Stacking集成学习器
        
        完整流程：
        1. 训练所有基模型
        2. 生成元特征（基模型的验证集预测）
        3. 训练元学习器
        
        Args:
            X_train: 训练集特征 (n_samples, n_features)
            y_train: 训练集标签 (n_samples,)
            X_val: 验证集特征 (n_samples, n_features) - 用于生成元特征
            y_val: 验证集标签 (n_samples,)
            X_test: 测试集特征（可选）
            y_test: 测试集标签（可选）
            train_kwargs: 传递给基模型的训练参数
            **kwargs: 其他参数
        
        Returns:
            训练结果字典
        """
        logger.info("开始Stacking集成学习训练")
        logger.info(f"✓ 第1层：训练{len(self.base_models)}个基模型...")
        
        if train_kwargs is None:
            train_kwargs = {}
        
        # ============================================================
        # 第1层：训练基模型
        # ============================================================
        meta_features_train = []
        meta_features_val = []
        meta_features_test = [] if X_test is not None else None
        
        base_model_names = []
        
        for i, model in enumerate(self.base_models):
            logger.info(f"  [{i+1}/{len(self.base_models)}] 训练基模型：{model.model_name}")
            
            # 训练模型
            model.train(X_train, y_train, X_val, y_val, **train_kwargs)
            base_model_names.append(model.model_name)
            
            # 生成验证集预测（元特征）
            y_val_pred = model.predict(X_val)
            meta_features_val.append(y_val_pred.reshape(-1, 1))
            logger.info(f"    验证集预测完成。形状: {y_val_pred.shape}")
            
            # 生成测试集预测（如果提供）
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                meta_features_test.append(y_test_pred.reshape(-1, 1))
        
        # 合并元特征
        meta_features_val = np.hstack(meta_features_val)
        logger.info(f"✓ 元特征生成完成。形状: {meta_features_val.shape}")
        
        if X_test is not None:
            meta_features_test = np.hstack(meta_features_test)
            logger.info(f"  测试集元特征形状: {meta_features_test.shape}")
        
        # ============================================================
        # 第2层：训练元学习器
        # ============================================================
        logger.info(f"✓ 第2层：训练元学习器（{self.meta_learner_type}）...")
        
        self._fit_meta_learner(meta_features_val, y_val)
        
        # ============================================================
        # 第3层：评估
        # ============================================================
        logger.info(f"✓ 第3层：评估集成模型...")
        
        # 验证集评估
        y_val_pred_meta = self.meta_learner.predict(meta_features_val)
        val_metrics = self._evaluate_predictions(y_val, y_val_pred_meta, 'validation')
        
        # 测试集评估（如果提供）
        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred_meta = self.meta_learner.predict(meta_features_test)
            test_metrics = self._evaluate_predictions(y_test, y_test_pred_meta, 'test')
        
        self.is_fitted = True
        self.training_history = {
            'base_models': base_model_names,
            'meta_learner': self.meta_learner_type,
            'meta_features_shape': meta_features_val.shape,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Stacking集成学习训练完成！")
        logger.info(f"  验证集 - MAE: {val_metrics['mae']:.6f}, RMSE: {val_metrics['rmse']:.6f}")
        
        return self.training_history
    
    def predict(self, X_test: np.ndarray, return_base_predictions: bool = False) -> np.ndarray | Tuple:
        """
        进行预测
        
        Args:
            X_test: 测试特征 (n_samples, n_features)
            return_base_predictions: 是否返回基模型预测
        
        Returns:
            集成预测结果，或 (集成预测, 基模型预测字典)
        """
        if not self.is_fitted:
            raise RuntimeError("Stacking模型未训练，请先调用 fit() 方法")
        
        logger.info(f"开始Stacking预测。测试集形状: {X_test.shape}")
        
        # 生成元特征
        meta_features = []
        base_predictions = {}
        
        for model in self.base_models:
            y_pred = model.predict(X_test)
            meta_features.append(y_pred.reshape(-1, 1))
            base_predictions[model.model_name] = y_pred
        
        meta_features = np.hstack(meta_features)
        
        # 使用元学习器预测
        y_pred_final = self.meta_learner.predict(meta_features)
        
        logger.info(f"Stacking预测完成。预测结果形状: {y_pred_final.shape}")
        
        if return_base_predictions:
            return y_pred_final, base_predictions
        else:
            return y_pred_final
    
    def save(self, path: str = None) -> str:
        """
        保存集成模型（所有基模型+元学习器）
        
        Args:
            path: 保存路径（默认为 models/stacking_model.pkl）
        
        Returns:
            实际保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无法保存")
        
        if path is None:
            path = self.model_dir / 'stacking_model.pkl'
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存所有内容
        checkpoint = {
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'meta_learner_type': self.meta_learner_type,
            'training_history': self.training_history,
            'final_metrics': self.final_metrics
        }
        
        joblib.dump(checkpoint, path)
        logger.info(f"Stacking模型已保存到：{path}")
        
        # 同时保存各个基模型的详细信息
        for model in self.base_models:
            model_path = self.model_dir / f"{model.model_name.lower()}_base.pth"
            model.save(str(model_path))
        
        return str(path)
    
    def load(self, path: str) -> StackingTrainer:
        """
        加载集成模型
        
        Args:
            path: 加载路径
        
        Returns:
            加载后的StackingTrainer实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        
        checkpoint = joblib.load(path)
        self.base_models = checkpoint['base_models']
        self.meta_learner = checkpoint['meta_learner']
        self.meta_learner_type = checkpoint['meta_learner_type']
        self.training_history = checkpoint['training_history']
        self.is_fitted = True
        
        logger.info(f"Stacking模型已加载：{path}")
        return self
    
    def _fit_meta_learner(self, X_meta: np.ndarray, y_train: np.ndarray):
        """
        训练元学习器
        
        Args:
            X_meta: 元特征 (n_samples, n_base_models)
            y_train: 目标标签 (n_samples,)
        """
        if self.meta_learner_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_learner = LinearRegression()
        
        elif self.meta_learner_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        
        elif self.meta_learner_type == 'xgboost':
            try:
                import xgboost as xgb
                self.meta_learner = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
            except ImportError:
                logger.warning("XGBoost未安装，使用Ridge替代")
                from sklearn.linear_model import Ridge
                self.meta_learner = Ridge()
        
        else:
            raise ValueError(f"未知的元学习器类型：{self.meta_learner_type}")
        
        logger.info(f"开始训练元学习器：{self.meta_learner_type}")
        self.meta_learner.fit(X_meta, y_train)
        logger.info(f"元学习器训练完成")
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                            dataset_name: str = 'test') -> Dict[str, float]:
        """
        评估预测性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dataset_name: 数据集名称（用于日志）
        
        Returns:
            指标字典
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        logger.info(f"{dataset_name.capitalize()}集 - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'base_models': [m.model_name for m in self.base_models],
            'meta_learner': self.meta_learner_type,
            'training_history': self.training_history
        }
    
    def __repr__(self) -> str:
        models_str = ', '.join([m.model_name for m in self.base_models])
        return (f"StackingTrainer(base_models=[{models_str}], "
                f"meta_learner={self.meta_learner_type}, fitted={self.is_fitted})")
