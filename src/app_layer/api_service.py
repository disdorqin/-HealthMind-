"""API服务"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.core.utils.logger import logger
from src.core.utils.training_progress import get_training_tracker


def create_api_app() -> Flask:
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)
    
    # 应用配置
    app.config['JSON_AS_ASCII'] = False
    app.config['MODEL_PATH'] = 'models/lstm_forecaster.pth'
    app.config['DATA_PATH'] = 'data/data.csv'
    
    # 注册蓝图（如果有的话）
    _register_error_handlers(app)
    _register_routes(app)
    
    logger.info("Flask应用创建成功")
    
    return app


def _register_error_handlers(app: Flask):
    """注册错误处理器"""
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'status': 'error', 'message': '请求参数错误'}), 400
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'status': 'error', 'message': '接口不存在'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"服务器错误：{str(e)}")
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500


def _register_routes(app: Flask):
    """注册路由"""
    
    @app.route('/api/health', methods=['GET'])
    def health_check() -> Dict[str, Any]:
        """健康检查"""
        return jsonify({
            'status': 'ok',
            'message': 'API服务运行正常',
            'service': 'Power Prediction System'
        })
    
    @app.route('/api/version', methods=['GET'])
    def get_version() -> Dict[str, Any]:
        """获取版本信息"""
        return jsonify({
            'status': 'ok',
            'version': '1.0.0',
            'name': '风芒可测 - 电力预测与交易优化系统'
        })
    
    @app.route('/api/run_full_pipeline', methods=['POST'])
    def run_full_pipeline() -> Dict[str, Any]:
        """
        运行完整管道
        请求体：{
            "data_path": "path/to/data.csv",
            "epochs": 50,
            "batch_size": 32,
            ...other_params
        }
        """
        try:
            params = request.get_json() or {}
            logger.info(f"开始运行完整管道，参数：{params}")
            
            # 这里会被main.py中的run_full_pipeline()调用
            from src.runner.pipeline_router import run_pipeline
            
            # 运行训练
            train_result = run_pipeline('train_lstm', {
                'data_path': params.get('data_path', 'data/data.csv'),
                'model_path': params.get('model_path', 'models/lstm_forecaster.pth'),
                'epochs': params.get('epochs', 50),
                'hidden_dim': params.get('hidden_dim', 64),
                'num_layers': params.get('num_layers', 2),
                'batch_size': params.get('batch_size', 32),
                'lookback': params.get('lookback', 24),
            })
            
            # 运行预测
            predict_result = run_pipeline('predict_lstm', {
                'data_path': params.get('data_path', 'data/data.csv'),
                'model_path': params.get('model_path', 'models/lstm_forecaster.pth'),
            })
            
            logger.info("完整管道运行成功")
            
            return jsonify({
                'status': 'success',
                'message': '完整管道执行成功',
                'training': train_result,
                'prediction': {
                    'count': predict_result.get('count'),
                    'min': predict_result.get('min'),
                    'max': predict_result.get('max'),
                    'mean': predict_result.get('mean'),
                }
            })
        
        except Exception as e:
            logger.error(f"管道执行失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'管道执行失败：{str(e)}'
            }), 500
    
    @app.route('/api/train', methods=['POST'])
    def train_model() -> Dict[str, Any]:
        """训练模型"""
        try:
            params = request.get_json() or {}
            logger.info("开始训练模型")
            
            from src.runner.pipeline_router import run_pipeline
            
            result = run_pipeline('train_lstm', {
                'data_path': params.get('data_path', 'data/data.csv'),
                'model_path': params.get('model_path', 'models/lstm_forecaster.pth'),
                'epochs': params.get('epochs', 50),
                'hidden_dim': params.get('hidden_dim', 64),
                'num_layers': params.get('num_layers', 2),
                'batch_size': params.get('batch_size', 32),
                'lookback': params.get('lookback', 24),
            })
            
            return jsonify({
                'status': 'success',
                'message': '模型训练完成',
                'result': result
            })
        
        except Exception as e:
            logger.error(f"训练失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'训练失败：{str(e)}'
            }), 500
    
    @app.route('/api/predict', methods=['POST'])
    def predict() -> Dict[str, Any]:
        """预测"""
        try:
            params = request.get_json() or {}
            logger.info("开始预测")
            
            from src.runner.pipeline_router import run_pipeline
            
            result = run_pipeline('predict_lstm', {
                'data_path': params.get('data_path', 'data/data.csv'),
                'model_path': params.get('model_path', 'models/lstm_forecaster.pth'),
            })
            
            return jsonify({
                'status': 'success',
                'message': '预测完成',
                'result': {
                    'count': result.get('count'),
                    'min': result.get('min'),
                    'max': result.get('max'),
                    'mean': result.get('mean'),
                }
            })
        
        except Exception as e:
            logger.error(f"预测失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'预测失败：{str(e)}'
            }), 500
    
    # ============================================================
    # 交易优化相关API
    # ============================================================
    
    @app.route('/api/trade/advice', methods=['GET', 'POST'])
    def get_trade_advice() -> Dict[str, Any]:
        """
        获取交易建议 - 后端完整业务逻辑
        
        返回24小时的买入/卖出建议，基于：
        - 预测的功率值
        - 分时电价（高峰/平段/低谷）
        - 经济优化模型
        
        返回格式：
        {
            'status': 'success',
            'data': {
                'buy_advice': [...],    # 买入建议列表
                'sell_advice': [...],   # 卖出建议列表
                'summary': {...},       # 汇总信息（电价等）
                'expected_revenue': ...,  # 预期收益
                'cost_saving': ...,       # 成本节约
                'peak_shaving_power': ..., # 削峰功率
                'valley_filling_power': ..., # 填谷功率
            }
        }
        """
        try:
            logger.info("获取交易建议")
            
            from src.runner.pipeline_router import run_pipeline
            from src.app_layer.trade_service import TradeOptimizer, TimeOfUsePrice
            
            # 获取最新预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': 'data/data.csv',
                'model_path': 'models/lstm_forecaster.pth',
            })
            
            # 提取24小时预测值（如果不足24小时，补充）
            predictions = predict_result.get('predictions', [])
            if len(predictions) < 24:
                # 补充到24小时（使用最后一个值或均值）
                avg_pred = np.mean(predictions) if predictions else 500
                predictions = list(predictions) + [avg_pred] * (24 - len(predictions))
            predictions = np.array(predictions[:24])  # 只取前24小时
            
            # 创建交易优化器
            price_data = TimeOfUsePrice(
                peak_price=1.2,
                flat_price=0.8,
                valley_price=0.4
            )
            optimizer = TradeOptimizer(predictions, price_data)
            
            # 生成交易建议
            advice_result = optimizer.generate_trade_advice()
            
            logger.info("✓ 交易建议生成成功")
            
            return jsonify({
                'status': 'success',
                'data': advice_result
            })
        
        except Exception as e:
            logger.error(f"交易建议生成失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'交易建议生成失败：{str(e)}'
            }), 500
    
    @app.route('/api/trade/metrics', methods=['GET'])
    def get_trade_metrics() -> Dict[str, Any]:
        """
        获取交易指标 - 计算收益相关指标
        
        返回：
        {
            'status': 'success',
            'data': {
                'daily_energy': ...,               # 日发电量 (kWh)
                'average_cost_per_kwh': ...,       # 平均成本 (¥/kWh)
                'revenue_potential': ...,          # 收益潜力 (¥)
                'peak_shaving_benefit': ...,       # 削峰效益 (¥)
                'valley_filling_benefit': ...,     # 填谷效益 (¥)
                'total_benefit': ...,              # 总效益 (¥)
            }
        }
        """
        try:
            logger.info("获取交易指标")
            
            from src.runner.pipeline_router import run_pipeline
            from src.app_layer.trade_service import TradeOptimizer, TimeOfUsePrice
            
            # 获取预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': 'data/data.csv',
                'model_path': 'models/lstm_forecaster.pth',
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
            
            return jsonify({
                'status': 'success',
                'data': metrics
            })
        
        except Exception as e:
            logger.error(f"交易指标计算失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'交易指标计算失败：{str(e)}'
            }), 500
    
    @app.route('/api/trade/risk', methods=['GET'])
    def get_trade_risk() -> Dict[str, Any]:
        """
        交易风险分析 - 评估操作风险
        
        返回：
        {
            'status': 'success',
            'data': {
                'risk_score': ...,          # 风险分数 (0-10)
                'risk_level': 'LOW/MEDIUM/HIGH/...',  # 风险等级
                'recommendation': '...',    # 操作建议
                'statistics': {             # 统计信息
                    'mean': ...,
                    'std': ...,
                    'min': ...,
                    'max': ...,
                    ...
                },
                'risk_indicators': {        # 风险指标
                    'volatility_score': ...,
                    'range_score': ...,
                    ...
                }
            }
        }
        """
        try:
            logger.info("进行交易风险分析")
            
            from src.runner.pipeline_router import run_pipeline
            from src.app_layer.trade_service import TradeOptimizer, TimeOfUsePrice
            
            # 获取预测数据
            predict_result = run_pipeline('predict_lstm', {
                'data_path': 'data/data.csv',
                'model_path': 'models/lstm_forecaster.pth',
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
            
            return jsonify({
                'status': 'success',
                'data': risk_result
            })
        
        except Exception as e:
            logger.error(f"风险分析失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'风险分析失败：{str(e)}'
            }), 500
    
    # ============================================================
    # 数据和模型状态检查API
    # ============================================================
    
    @app.route('/api/data/status', methods=['GET'])
    def get_data_status() -> Dict[str, Any]:
        """
        获取数据文件状态
        
        返回：
        {
            'status': 'ok/error',
            'message': '...',
            'size': '文件大小 MB',
            'rows': '行数',
            'columns': '列数',
            'last_updated': '最后更新时间'
        }
        """
        try:
            from pathlib import Path
            import os
            from datetime import datetime
            
            data_path = 'data/data.csv'
            
            if not os.path.exists(data_path):
                return jsonify({
                    'status': 'error',
                    'message': '数据文件不存在'
                }), 404
            
            # 获取文件信息
            file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(data_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            # 读取行数和列数
            import pandas as pd
            try:
                df = pd.read_csv(data_path, nrows=1000)  # 只读前1000行获取统计信息
                total_rows = len(df)
                num_columns = len(df.columns)
                
                return jsonify({
                    'status': 'ok',
                    'message': '数据文件正常',
                    'size': f'{file_size:.2f}',
                    'rows': total_rows,
                    'columns': num_columns,
                    'last_updated': mod_time
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'无法读取数据文件：{str(e)}'
                }), 500
        
        except Exception as e:
            logger.error(f"获取数据状态失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'获取数据状态失败：{str(e)}'
            }), 500
    
    @app.route('/api/model/status', methods=['GET'])
    def get_model_status() -> Dict[str, Any]:
        """
        获取模型文件状态
        
        返回：
        {
            'status': 'ok/incomplete',
            'message': '...',
            'models': {
                'lstm_forecaster.pth': 'size MB',
                'preprocess_pipeline.joblib': 'size MB',
                ...
            }
        }
        """
        try:
            from pathlib import Path
            import os
            
            models_dir = 'models'
            
            if not os.path.exists(models_dir):
                return jsonify({
                    'status': 'incomplete',
                    'message': '模型目录不存在',
                    'models': {}
                })
            
            # 检查模型文件
            expected_models = [
                'lstm_forecaster.pth',
                'preprocess_pipeline.joblib'
            ]
            
            models_info = {}
            for model_file in os.listdir(models_dir):
                file_path = os.path.join(models_dir, model_file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    models_info[model_file] = f'{file_size:.2f} MB'
            
            status = 'ok' if len(models_info) >= len(expected_models) else 'incomplete'
            
            return jsonify({
                'status': status,
                'message': '模型就绪' if status == 'ok' else '部分模型缺失，建议重新训练',
                'models': models_info
            })
        
        except Exception as e:
            logger.error(f"获取模型状态失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'获取模型状态失败：{str(e)}'
            }), 500
    
    # ============================================================
    # 训练进度相关API
    # ============================================================
    
    @app.route('/api/training/progress', methods=['GET'])
    def get_training_progress() -> Dict[str, Any]:
        """
        获取训练进度
        
        返回：
        {
            'status': 'success/error',
            'data': {
                'task_id': '任务ID',
                'model_name': '模型名称',
                'total_epochs': '总轮数',
                'current_epoch': '当前轮次',
                'progress': '进度百分比',
                'status': 'running/completed/failed',
                'loss_history': [{'epoch': ..., 'loss': ..., 'timestamp': ...}, ...],
                'metrics': {},
                'start_time': '开始时间戳'
            }
        }
        """
        try:
            tracker = get_training_tracker()
            # 从查询参数获取任务ID，如果没有则返回当前任务
            task_id = request.args.get('task_id', None)
            if not task_id:
                # 如果没有指定任务ID，返回当前正在跟踪的任务
                progress_data = tracker._load_progress()  # 使用私有方法获取当前进度
                if not progress_data:
                    return jsonify({
                        'status': 'success',
                        'message': '无正在进行的训练任务',
                        'data': None
                    })
                return jsonify({
                    'status': 'success',
                    'data': progress_data
                })
            else:
                progress_data = tracker.get_progress(task_id)
                if progress_data:
                    return jsonify({
                        'status': 'success',
                        'data': progress_data
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'message': '未找到指定任务的进度信息',
                        'data': None
                    })
        except Exception as e:
            logger.error(f"获取训练进度失败：{str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'获取训练进度失败：{str(e)}'
            }), 500
