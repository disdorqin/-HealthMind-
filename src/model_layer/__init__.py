"""模型层 - 负责模型训练和推理"""

# 基础组件
from .base_model import BaseModel
from .lstm_model import LSTMModel
from .model_evaluator import ModelEvaluator

# 新增模型包装
from .lstm_wrapper import LSTMModelWrapper
from .xgboost_model import XGBoostModel, XGBoostClassifier

# Stacking集成学习
from .stacking_trainer import StackingTrainer

# 高级工具
from .metrics_manager import ModelMetricsManager
from .training_pipeline import ModelTrainingPipeline

# 原有组件（保持向后兼容）
from .model_trainer import MultiModelTrainer

__all__ = [
    # 基础
    'BaseModel',
    'LSTMModel',
    'ModelEvaluator',
    
    # 模型包装
    'LSTMModelWrapper',
    'XGBoostModel',
    'XGBoostClassifier',
    
    # 集成学习
    'StackingTrainer',
    
    # 工具和管道
    'ModelMetricsManager',
    'ModelTrainingPipeline',
    
    # 原有
    'MultiModelTrainer',
]
