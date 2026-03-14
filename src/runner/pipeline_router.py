"""管道路由器 - 负责调度和执行不同管道"""

from __future__ import annotations

from typing import Any, Dict

from src.pipeline.lstm_pipeline import LSTMPipeline, LSTMPipelineResult

from src.core.utils.training_progress import get_training_tracker


class PipelineRouter:
    """管道路由器"""
    
    def __init__(self):
        self.pipelines = {
            'lstm': LSTMPipeline,
        }
    
    def run(self, pipeline_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """运行指定管道"""
        params = params or {}
        
        if pipeline_name == 'train_lstm':
            return self._train_lstm(params)
        elif pipeline_name == 'predict_lstm':
            return self._predict_lstm(params)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    def _train_lstm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """训练 LSTM 模型"""
        pipeline = LSTMPipeline(
            lookback=params.get('lookback', 24),
            hidden_dim=params.get('hidden_dim', 16),
            num_layers=params.get('num_layers', 1),
            epochs=params.get('epochs', 10),
            batch_size=params.get('batch_size', 256),
        )
        
        result = pipeline.train(
            data_path=params.get('data_path', 'data/data.csv'),
            model_save_path=params.get('model_path', 'models/lstm_forecaster.pth'),
            nrows=params.get('nrows'),
        )
        
        return result.to_dict()
    
    def _predict_lstm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """LSTM 预测"""
        pipeline = LSTMPipeline()
        
        predictions = pipeline.predict(
            data_path=params.get('data_path', 'data/data.csv'),
            model_path=params.get('model_path', 'models/lstm_forecaster.pth'),
            nrows=params.get('nrows'),
        )
        
        return {
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'mean': float(predictions.mean()),
        }


# 便捷函数
def run_pipeline(pipeline_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """运行管道的便捷函数"""
    router = PipelineRouter()
    return router.run(pipeline_name, params)