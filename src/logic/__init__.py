"""业务逻辑层统一接口 - Streamlit 应用直接调用"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from src.core.utils.logger import logger
from src.runner.pipeline_router import run_pipeline
from src.logic.trade import TradeOptimizer, TimeOfUsePrice


class BusinessLogic:
    """业务逻辑聚合器 - Streamlit 应用的核心业务接口"""
    
    @staticmethod
    def run_full_pipeline(data_path: str = 'data/data.csv', 
                          model_path: str = 'models/lstm_forecaster.pth',
                          epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """运行完整管道（数据导入->模型训练->预测）"""
        logger.info("运行完整管道")
        
        try:
            # 训练模型
            train_result = run_pipeline('train_lstm', {
                'data_path': data_path,
                'model_path': model_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'hidden_dim': 64,
                'num_layers': 2,
                'lookback': 24,
            })
            
            # 运行预测
            predict_result = run_pipeline('predict_lstm', {
                'data_path': data_path,
                'model_path': model_path,
            })
            
            logger.info("✓ 完整管道执行成功")
            
            return {
                'status': 'success',
                'message': '完整管道执行成功',
                'training': train_result,
                'prediction': {
                    'count': predict_result.get('count'),
                    'min': predict_result.get('min'),
                    'max': predict_result.get('max'),
                    'mean': predict_result.get('mean'),
                }
            }
        except Exception as e:
            logger.error(f"管道执行失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'管道执行失败：{str(e)}'
            }
    
    @staticmethod
    def train_model(data_path: str = 'data/data.csv',
                   model_path: str = 'models/lstm_forecaster.pth',
                   epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """训练模型"""
        logger.info("开始训练模型")
        
        try:
            result = run_pipeline('train_lstm', {
                'data_path': data_path,
                'model_path': model_path,
                'epochs': epochs,
                'hidden_dim': 64,
                'num_layers': 2,
                'batch_size': batch_size,
                'lookback': 24,
            })
            
            return {
                'status': 'success',
                'message': '模型训练完成',
                'result': result
            }
        except Exception as e:
            logger.error(f"训练失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'训练失败：{str(e)}'
            }
    
    @staticmethod
    def predict(data_path: str = 'data/data.csv',
               model_path: str = 'models/lstm_forecaster.pth') -> Dict[str, Any]:
        """预测"""
        logger.info("开始预测")
        
        try:
            result = run_pipeline('predict_lstm', {
                'data_path': data_path,
                'model_path': model_path,
            })
            
            return {
                'status': 'success',
                'message': '预测完成',
                'result': {
                    'count': result.get('count'),
                    'min': result.get('min'),
                    'max': result.get('max'),
                    'mean': result.get('mean'),
                    'predictions': result.get('predictions'),
                }
            }
        except Exception as e:
            logger.error(f"预测失败：{str(e)}")
            return {
                'status': 'error',
                'message': f'预测失败：{str(e)}'
            }
    
    @staticmethod
    def get_trade_advice(data_path: str = 'data/data.csv',
                        model_path: str = 'models/lstm_forecaster.pth') -> Dict[str, Any]:
        """获取交易建议"""
        logger.info("获取交易建议")
        
        try:
            # 获取最新预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': data_path,
                'model_path': model_path,
            })
            
            # 提取24小时预测值
            predictions = predict_result.get('predictions', [])
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
            # 获取预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': data_path,
                'model_path': model_path,
            })
            
            predictions = predict_result.get('predictions', [])
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
            # 获取预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': data_path,
                'model_path': model_path,
            })
            
            predictions = predict_result.get('predictions', [])
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
