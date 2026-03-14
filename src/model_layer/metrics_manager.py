"""模型指标管理 - 自动保存训练指标到数据库"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path

from src.core.utils.logger import logger

try:
    from database.schema import TrainingMetricsModel
    from database.db_config import SessionLocal
    HAS_DB = True
except ImportError:
    HAS_DB = False


class ModelMetricsManager:
    """
    模型指标管理器 - 自动保存训练指标到数据库
    
    功能：
    - 保存模型训练指标（MAE, RMSE, MAPE, R² 等）
    - 支持离线模式（无数据库连接时）
    - 自动版本管理
    - 查询和比较历史指标
    
    示例：
        >>> metrics_mgr = ModelMetricsManager()
        >>> metrics_mgr.save_metrics(
        ...     model_name='LSTM',
        ...     mae=0.05,
        ...     rmse=0.08,
        ...     metrics={'r2': 0.95}
        ... )
    """
    
    def __init__(self, db_available: bool = True):
        """
        初始化指标管理器
        
        Args:
            db_available: 是否可用数据库（若为False则仅保存本地JSON）
        """
        self.db_available = db_available and HAS_DB
        
        if not self.db_available:
            logger.warning("数据库不可用。指标将仅保存为本地JSON文件")
        
        logger.info(f"模型指标管理器初始化 - DB可用: {self.db_available}")
    
    def save_metrics(self, model_name: str, mae: float, rmse: float,
                    model_type: str = 'Unknown',
                    mape: float = None, r2: float = None, mse: float = None,
                    epochs: int = None, batch_size: int = None,
                    learning_rate: float = None, training_time: float = None,
                    dataset_size: int = None,
                    validation_metrics: Dict[str, float] = None,
                    test_metrics: Dict[str, float] = None,
                    version: str = None, notes: str = None,
                    metrics: Dict[str, Any] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        保存模型训练指标到数据库和本地文件
        
        Args:
            model_name: 模型名称（如 'LSTM', 'XGBoost', 'Stacking'）
            mae: 平均绝对误差
            rmse: 均方根误差
            model_type: 模型类型（'DL', 'Tree', 'Ensemble' 等）
            mape: 平均绝对百分比误差
            r2: R² 评分
            mse: 均方误差
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            training_time: 训练耗时（秒）
            dataset_size: 数据集大小
            validation_metrics: 验证集指标字典
            test_metrics: 测试集指标字典
            version: 模型版本
            notes: 备注
            metrics: 其他指标字典（会与主要指标合并）
            **kwargs: 其他参数
        
        Returns:
            保存结果字典
        """
        logger.info(f"保存模型指标：{model_name}")
        
        # 如果提供了metrics字典，从中提取缺失的指标
        if metrics:
            mape = mape or metrics.get('mape')
            r2 = r2 or metrics.get('r2') or metrics.get('R2')
            mse = mse or metrics.get('mse') or metrics.get('MSE')
        
        # 准备数据库记录
        db_record = {
            'model_name': model_name,
            'model_type': model_type,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if mape is not None else None,
            'r2': float(r2) if r2 is not None else None,
            'mse': float(mse) if mse is not None else None,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': float(learning_rate) if learning_rate is not None else None,
            'training_time': float(training_time) if training_time is not None else None,
            'dataset_size': dataset_size,
            'validation_metrics': json.dumps(validation_metrics) if validation_metrics else None,
            'test_metrics': json.dumps(test_metrics) if test_metrics else None,
            'version': version,
            'notes': notes
        }
        
        result = {
            'status': 'success',
            'model_name': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'timestamp': datetime.now().isoformat()
        }
        
        # 尝试保存到数据库
        if self.db_available:
            try:
                self._save_to_database(db_record)
                result['database'] = 'saved'
            except Exception as e:
                logger.warning(f"数据库保存失败：{e}。将仅保存本地文件")
                result['database'] = f'failed: {str(e)}'
        else:
            result['database'] = 'not_available'
        
        # 保存本地JSON文件
        try:
            json_path = self._save_to_local_json(db_record)
            result['local_json'] = str(json_path)
            logger.info(f"✓ 指标已保存：{json_path}")
        except Exception as e:
            logger.error(f"本地保存失败：{e}")
            result['local_json'] = f'failed: {str(e)}'
        
        logger.info(f"指标保存完成：MAE={mae:.6f}, RMSE={rmse:.6f}")
        return result
    
    def _save_to_database(self, record: Dict[str, Any]):
        """
        保存指标到数据库
        
        Args:
            record: 指标记录字典
        """
        if not self.db_available:
            raise RuntimeError("数据库不可用")
        
        try:
            from database.db_config import SessionLocal
            
            session = SessionLocal()
            try:
                # 创建数据库记录对象
                metric = TrainingMetricsModel(**record)
                session.add(metric)
                session.commit()
                logger.info(f"✓ 指标已保存到数据库")
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"数据库操作失败：{e}")
            raise
    
    def _save_to_local_json(self, record: Dict[str, Any]) -> Path:
        """
        保存指标到本地JSON文件
        
        Args:
            record: 指标记录字典
        
        Returns:
            保存路径
        """
        # 创建results目录
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 文件名包含时间戳和模型名称
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"metrics_{record['model_name']}_{timestamp}.json"
        filepath = results_dir / filename
        
        # 转换JSON不可序列化的字段
        record_copy = record.copy()
        record_copy['timestamp'] = datetime.now().isoformat()
        
        # 写入JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record_copy, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def query_model_metrics(self, model_name: str, limit: int = 10) -> list:
        """
        查询特定模型的历史指标
        
        Args:
            model_name: 模型名称
            limit: 返回记录数上限
        
        Returns:
            指标列表
        """
        if not self.db_available:
            logger.warning("数据库不可用，无法查询")
            return []
        
        try:
            session = SessionLocal()
            try:
                records = (
                    session.query(TrainingMetricsModel)
                    .filter(TrainingMetricsModel.model_name == model_name)
                    .order_by(TrainingMetricsModel.run_time.desc())
                    .limit(limit)
                    .all()
                )
                
                return [r.to_dict() for r in records]
            
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"查询失败：{e}")
            return []
    
    def compare_models(self, metric: str = 'mae') -> list:
        """
        比较所有模型的最新指标
        
        Args:
            metric: 比较指标（'mae', 'rmse', 'r2' 等）
        
        Returns:
            按指标值排序的模型列表
        """
        if not self.db_available:
            logger.warning("数据库不可用，无法比较")
            return []
        
        try:
            from sqlalchemy import func
            
            session = SessionLocal()
            try:
                # 获取每个模型最新的指标
                subquery = (
                    session.query(
                        TrainingMetricsModel.model_name,
                        func.max(TrainingMetricsModel.run_time).label('max_run_time')
                    )
                    .group_by(TrainingMetricsModel.model_name)
                    .subquery()
                )
                
                records = (
                    session.query(TrainingMetricsModel)
                    .join(
                        subquery,
                        (TrainingMetricsModel.model_name == subquery.c.model_name) &
                        (TrainingMetricsModel.run_time == subquery.c.max_run_time)
                    )
                    .all()
                )
                
                # 按指标排序
                if metric.lower() in ['mae', 'rmse', 'mape']:
                    # 这些指标越小越好
                    records.sort(key=lambda x: getattr(x, metric) or float('inf'))
                else:
                    # r2等指标越大越好
                    records.sort(key=lambda x: getattr(x, metric) or 0, reverse=True)
                
                return [r.to_dict() for r in records]
            
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"比较失败：{e}")
            return []
    
    def get_best_model(self, metric: str = 'mae') -> Optional[Dict[str, Any]]:
        """
        获取最优模型（按指定指标）
        
        Args:
            metric: 评估指标（'mae', 'rmse', 'r2' 等）
        
        Returns:
            最优模型的指标字典，或None
        """
        comparison = self.compare_models(metric)
        
        if comparison:
            logger.info(f"最优模型（按{metric}）：{comparison[0]['model_name']}")
            return comparison[0]
        
        return None
    
    def generate_report(self) -> str:
        """生成所有模型的性能报告"""
        if not self.db_available:
            return "数据库不可用，无法生成报告"
        
        comparison = self.compare_models('mae')
        
        if not comparison:
            return "暂无训练记录"
        
        report = "\n" + "="*70 + "\n"
        report += "模型训练指标报告\n"
        report += "="*70 + "\n\n"
        
        for i, model in enumerate(comparison, 1):
            report += f"{i}. {model['model_name']} ({model['model_type']})\n"
            report += f"   训练时间：{model['run_time']}\n"
            report += f"   MAE: {model['mae']:.6f}\n"
            report += f"   RMSE: {model['rmse']:.6f}\n"
            if model.get('r2'):
                report += f"   R²: {model['r2']:.6f}\n"
            if model.get('mape'):
                report += f"   MAPE: {model['mape']:.2f}%\n"
            if model.get('notes'):
                report += f"   备注：{model['notes']}\n"
            report += "\n"
        
        report += "="*70 + "\n"
        
        return report
