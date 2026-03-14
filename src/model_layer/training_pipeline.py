"""统一的模型训练管道 - 一键完整训练流程"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

import numpy as np

from src.core.utils.logger import logger
from .base_model import BaseModel
from .stacking_trainer import StackingTrainer
from .metrics_manager import ModelMetricsManager


class ModelTrainingPipeline:
    """
    统一的模型训练管道 - 完整的训练、评估、保存流程
    
    功能：
    1. 单模型训练：训练单个模型（LSTM、XGBoost等）
    2. Stacking集成学习：多个基模型 + 元学习器
    3. 自动模型保存：将训练好的模型保存到 /models
    4. 自动指标保存：将评价指标保存到数据库和本地JSON
    5. 格式标准化：统一的输入输出接口
    
    示例：
        # 方式1：单模型训练
        >>> pipeline = ModelTrainingPipeline('LSTM')
        >>> result = pipeline.train(X_train, y_train, X_val, y_val)
        >>> pipeline.save_model('models/lstm_model.pkl')
        
        # 方式2：Stacking集成学习
        >>> lstm_model = LSTMModelWrapper(input_dim=8)
        >>> xgb_model = XGBoostModel(n_estimators=100)
        >>> pipeline = ModelTrainingPipeline('Stacking', models=[lstm_model, xgb_model])
        >>> result = pipeline.train(X_train, y_train, X_val, y_val, X_test, y_test)
    """
    
    def __init__(self, model_type: str = 'LSTM', models: List[BaseModel] = None,
                 model_dir: str = 'models', auto_save: bool = True,
                 auto_db_save: bool = True):
        """
        初始化训练管道
        
        Args:
            model_type: 模型类型（'LSTM', 'XGBoost', 'Stacking' 等）
            models: 模型列表（用于Stacking，应为BaseModel实例）
            model_dir: 模型保存目录
            auto_save: 是否自动保存模型到文件
            auto_db_save: 是否自动保存指标到数据库
        """
        self.model_type = model_type
        self.model = None
        self.models = models
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.auto_db_save = auto_db_save
        
        self.training_result = None
        self.metrics = {}
        self.predictions = {}
        self.training_time = 0
        
        # 初始化指标管理器
        self.metrics_manager = ModelMetricsManager(db_available=auto_db_save)
        
        logger.info(f"模型训练管道初始化 - model_type: {model_type}, auto_save: {auto_save}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              X_test: Optional[np.ndarray] = None,
              y_test: Optional[np.ndarray] = None,
              train_kwargs: Optional[Dict[str, Any]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型（支持单模型和Stacking）
        
        Args:
            X_train: 训练集特征 (n_samples, n_features)
            y_train: 训练集标签 (n_samples,)
            X_val: 验证集特征
            y_val: 验证集标签
            X_test: 测试集特征（可选）
            y_test: 测试集标签（可选）
            train_kwargs: 传递给模型的训练参数
            **kwargs: 其他参数
        
        Returns:
            训练结果字典 {'status': 'success', 'metrics': {...}, ...}
        """
        logger.info(f"开始训练模型：{self.model_type}")
        start_time = time.time()
        
        if train_kwargs is None:
            train_kwargs = {}
        
        try:
            if self.model_type.lower() == 'stacking':
                # Stacking集成学习
                result = self._train_stacking(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    train_kwargs, **kwargs
                )
            else:
                # 单模型训练
                result = self._train_single_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    train_kwargs, **kwargs
                )
            
            self.training_time = time.time() - start_time
            result['training_time'] = self.training_time
            self.training_result = result
            
            # 自动保存模型和指标
            if self.auto_save:
                model_path = self._auto_save_model()
                result['model_path'] = model_path
            
            if self.auto_db_save:
                db_result = self._auto_save_metrics()
                result['db_save'] = db_result
            
            logger.info(f"✅ 模型训练完成！耗时 {self.training_time:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f"模型训练失败：{e}")
            raise
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, X_test, y_test,
                           train_kwargs, **kwargs) -> Dict[str, Any]:
        """训练单个模型"""
        from .lstm_wrapper import LSTMModelWrapper
        from .xgboost_model import XGBoostModel
        
        # 创建模型
        if self.model is None:
            if self.model_type.lower() == 'lstm':
                input_dim = X_train.shape[1] if len(X_train.shape) > 1 else 1
                self.model = LSTMModelWrapper(input_dim=input_dim, **kwargs)
            elif self.model_type.lower() == 'xgboost':
                self.model = XGBoostModel(**kwargs)
            else:
                raise ValueError(f"未知的模型类型：{self.model_type}")
        
        # 训练
        logger.info(f"训练{self.model.model_name}模型")
        self.model.train(X_train, y_train, X_val, y_val, **train_kwargs)
        
        # 预测
        y_val_pred = self.model.predict(X_val)
        self.predictions['val'] = y_val_pred
        
        # 评估验证集
        val_metrics = self.model.evaluate(y_val, y_val_pred)
        
        # 评估测试集（如果提供）
        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            self.predictions['test'] = y_test_pred
            test_metrics = self.model.evaluate(y_test, y_test_pred)
        
        self.metrics = val_metrics
        
        result = {
            'status': 'success',
            'model_type': self.model_type,
            'model_name': self.model.model_name,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_history': self.model.training_history
        }
        
        return result
    
    def _train_stacking(self, X_train, y_train, X_val, y_val, X_test, y_test,
                       train_kwargs, **kwargs) -> Dict[str, Any]:
        """训练Stacking集成模型"""
        if not self.models:
            raise ValueError("Stacking需要提供基模型列表")
        
        # 创建Stacking训练器
        meta_learner = kwargs.get('meta_learner', 'linear')
        stacking = StackingTrainer(self.models, meta_learner=meta_learner,
                                  model_dir=str(self.model_dir))
        
        # 训练
        logger.info("开始Stacking集成学习")
        fit_result = stacking.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                                 train_kwargs=train_kwargs)
        
        # 预测
        y_val_pred = stacking.predict(X_val)
        self.predictions['val'] = y_val_pred
        
        # 评估
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        val_metrics = {
            'mae': float(val_mae),
            'rmse': float(val_rmse),
            'r2': float(val_r2)
        }
        
        self.metrics = val_metrics
        self.model = stacking  # 保存stacking对象
        
        result = {
            'status': 'success',
            'model_type': 'Stacking',
            'model_name': 'Stacking',
            'val_metrics': val_metrics,
            'test_metrics': fit_result.get('test_metrics'),
            'training_history': fit_result
        }
        
        return result
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        进行预测
        
        Args:
            X_test: 测试特征
        
        Returns:
            预测值
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        return self.model.predict(X_test)
    
    def _auto_save_model(self) -> str:
        """自动保存模型到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.model_type.lower() == 'stacking':
            model_path = self.model_dir / f"stacking_{timestamp}.pkl"
        else:
            model_path = self.model_dir / f"{self.model_type.lower()}_{timestamp}.pth"
        
        self.model.save(str(model_path))
        logger.info(f"✓ 模型已自动保存到：{model_path}")
        
        return str(model_path)
    
    def _auto_save_metrics(self) -> Dict[str, Any]:
        """自动保存指标到数据库"""
        if not self.metrics:
            logger.warning("没有可保存的指标")
            return {'status': 'no_metrics'}
        
        # 准备保存参数
        save_params = {
            'model_name': self.model_type,
            'model_type': 'DL' if self.model_type.lower() == 'lstm' else \
                         'Tree' if self.model_type.lower() == 'xgboost' else 'Ensemble',
            'mae': self.metrics.get('mae'),
            'rmse': self.metrics.get('rmse'),
            'training_time': self.training_time,
            'version': datetime.now().strftime('%Y%m%d'),
        }
        
        # 添加其他可用的指标
        if 'mape' in self.metrics:
            save_params['mape'] = self.metrics['mape']
        if 'r2' in self.metrics:
            save_params['r2'] = self.metrics['r2']
        if 'mse' in self.metrics:
            save_params['mse'] = self.metrics['mse']
        
        # 保存指标
        result = self.metrics_manager.save_metrics(**save_params)
        logger.info(f"✓ 指标已自动保存")
        
        return result
    
    def save_model(self, path: str) -> str:
        """
        手动保存模型
        
        Args:
            path: 保存路径
        
        Returns:
            实际保存路径
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        return self.model.save(path)
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        if self.model is None:
            from .lstm_wrapper import LSTMModelWrapper
            self.model = LSTMModelWrapper(input_dim=1)
        
        self.model.load(path)
        logger.info(f"✓ 模型已加载：{path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'model_type': self.model_type,
            'status': 'trained' if self.model is not None else 'not_trained',
            'metrics': self.metrics,
            'training_time': self.training_time,
            'training_result': self.training_result
        }
    
    def get_metrics_report(self) -> str:
        """生成指标报告"""
        report = self.metrics_manager.generate_report()
        return report
    
    def __repr__(self) -> str:
        return f"ModelTrainingPipeline(model_type={self.model_type}, trained={self.model is not None})"
