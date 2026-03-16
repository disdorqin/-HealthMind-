"""Business logic facade, now backed by src.models.ModelService."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.utils.logger import logger
from src.logic.trade import TimeOfUsePrice, TradeOptimizer
from src.services import ForecastService


class BusinessLogic:
    """Compatibility facade used by app.py and main.py."""

    @staticmethod
    def _build_service(data_path: str = "data/data.csv") -> ForecastService:
        return ForecastService(data_path=data_path, model_dir="models", lookback=24)

    @staticmethod
    def run_full_pipeline(
        data_path: str = "data/data.csv",
        model_path: str = "models/lstm_forecaster.pth",
        epochs: int = 12,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        logger.info("Running full multi-model pipeline")
        try:
            service = BusinessLogic._build_service(data_path)
            selected_models = ["lstm", "gru", "xgboost", "moirai"]
            train_result = service.train(
                selected_models=selected_models,
                epochs=epochs,
                batch_size=batch_size,
            )
            predict_result = service.predict_by_horizon("day")

            stacking_values = predict_result.get("predictions_ensemble", [])
            values = np.asarray(stacking_values, dtype=np.float32)
            return {
                "status": "success",
                "message": "Full pipeline executed successfully",
                "training": train_result,
                "prediction": {
                    "count": int(values.size),
                    "min": float(values.min()) if values.size else 0.0,
                    "max": float(values.max()) if values.size else 0.0,
                    "mean": float(values.mean()) if values.size else 0.0,
                    "predictions": stacking_values,
                },
            }
        except Exception as exc:
            logger.error("Pipeline execution failed: %s", exc)
            return {"status": "error", "message": f"Pipeline execution failed: {exc}"}

    @staticmethod
    def train_model(
        data_path: str = "data/data.csv",
        model_path: str = "models/lstm_forecaster.pth",
        epochs: int = 12,
        batch_size: int = 128,
        selected_models: List[str] | None = None,
    ) -> Dict[str, Any]:
        logger.info("Start training models")
        try:
            service = BusinessLogic._build_service(data_path)
            model_list = selected_models or ["lstm", "gru", "xgboost", "moirai"]
            result = service.train(model_list, epochs=epochs, batch_size=batch_size)
            return {"status": "success", "message": "Model training completed", "result": result}
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return {"status": "error", "message": f"Training failed: {exc}"}

    @staticmethod
    def predict(
        data_path: str = "data/data.csv",
        model_path: str = "models/lstm_forecaster.pth",
        selected_models: List[str] | None = None,
    ) -> Dict[str, Any]:
        logger.info("Start prediction")
        try:
            service = BusinessLogic._build_service(data_path)
            result = service.predict_by_horizon("day")
            stacked = np.asarray(result.get("predictions_ensemble", []), dtype=np.float32)
            return {
                "status": "success",
                "message": "Prediction completed",
                "result": {
                    "count": int(stacked.size),
                    "min": float(stacked.min()) if stacked.size else 0.0,
                    "max": float(stacked.max()) if stacked.size else 0.0,
                    "mean": float(stacked.mean()) if stacked.size else 0.0,
                    "predictions": {**result.get("predictions", {}), "stacking": result.get("predictions_ensemble", [])},
                    "ground_truth": result.get("ground_truth", []),
                },
            }
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {"status": "error", "message": f"Prediction failed: {exc}"}
    
    @staticmethod
    def get_trade_advice(data_path: str = 'data/data.csv',
                        model_path: str = 'models/lstm_forecaster.pth') -> Dict[str, Any]:
        """获取交易建议"""
        logger.info("获取交易建议")
        
        try:
            # 获取最新预测数据（优先使用stacking）
            predict_result = BusinessLogic.predict(data_path=data_path, model_path=model_path)
            pred_payload = predict_result.get('result', {}).get('predictions', {})
            predictions = pred_payload.get('stacking') or pred_payload.get('lstm') or []
            if len(predictions) < 24:
                avg_pred = np.mean(predictions) if predictions else 500
                predictions = list(predictions) + [avg_pred] * (24 - len(predictions))
            predictions = np.array(predictions[:24])
            
            # 创建交易优化器
            price_data = TimeOfUsePrice()
            optimizer = TradeOptimizer(predictions, price_data)
            
            # 生成交易建议
            advice_result = optimizer.generate_trade_advice()
            
            logger.info("✓ 交易建议生成成功")
            
            return {
                'status': 'success',
                'data': advice_result
            }
        except Exception as e:
            logger.error(f"交易建议生成失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'交易建议生成失败：{str(e)}'
            }
    
    @staticmethod
    def get_trade_metrics(data_path: str = 'data/data.csv',
                         model_path: str = 'models/lstm_forecaster.pth') -> Dict[str, Any]:
        """获取交易指标"""
        logger.info("获取交易指标")
        
        try:
            # 获取预测数据（优先使用stacking）
            predict_result = BusinessLogic.predict(data_path=data_path, model_path=model_path)
            pred_payload = predict_result.get('result', {}).get('predictions', {})
            predictions = pred_payload.get('stacking') or pred_payload.get('lstm') or []
            if len(predictions) < 24:
                avg_pred = np.mean(predictions) if predictions else 500
                predictions = list(predictions) + [avg_pred] * (24 - len(predictions))
            predictions = np.array(predictions[:24])
            
            # 创建优化器并计算指标
            price_data = TimeOfUsePrice()
            optimizer = TradeOptimizer(predictions, price_data)
            metrics = optimizer.calculate_trade_metrics()
            
            logger.info("✓ 交易指标计算完成")
            
            return {
                'status': 'success',
                'data': metrics
            }
        except Exception as e:
            logger.error(f"交易指标计算失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'交易指标计算失败：{str(e)}'
            }
    
    @staticmethod
    def get_trade_risk(data_path: str = 'data/data.csv',
                      model_path: str = 'models/lstm_forecaster.pth') -> Dict[str, Any]:
        """获取交易风险分析"""
        logger.info("进行交易风险分析")
        
        try:
            # 获取预测数据（优先使用stacking）
            predict_result = BusinessLogic.predict(data_path=data_path, model_path=model_path)
            pred_payload = predict_result.get('result', {}).get('predictions', {})
            predictions = pred_payload.get('stacking') or pred_payload.get('lstm') or []
            if len(predictions) < 24:
                avg_pred = np.mean(predictions) if predictions else 500
                predictions = list(predictions) + [avg_pred] * (24 - len(predictions))
            predictions = np.array(predictions[:24])
            
            # 创建优化器并分析风险
            optimizer = TradeOptimizer(predictions)
            risk_result = optimizer.analyze_risk()
            
            logger.info("✓ 风险分析完成")
            
            return {
                'status': 'success',
                'data': risk_result
            }
        except Exception as e:
            logger.error(f"风险分析失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'风险分析失败：{str(e)}'
            }


# 导出接口
__all__ = ['BusinessLogic', 'TradeOptimizer', 'TimeOfUsePrice']
