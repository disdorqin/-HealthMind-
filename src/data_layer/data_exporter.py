"""优化的数据加载和导出模块"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import pandas as pd
import numpy as np
import joblib

from src.core.utils.logger import logger
from .data_schema import DataSchema
from .feature_engineering import FeatureEngineer
from .etl_pipeline import ETLPipeline
from .db_manager import db_manager


class DataLoader:
    """
    优化的数据加载器 - 支持CSV和数据库加载
    """
    
    def __init__(self, data_path: str = 'data/data.csv'):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
    
    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            nrows: 最多加载行数
        
        Returns:
            加载的数据
        """
        try:
            logger.info(f"加载数据：{self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
            
            self.data = pd.read_csv(self.data_path, nrows=nrows)
            logger.info(f"✓ 数据加载成功，形状：{self.data.shape}")
            
            return self.data
        
        except Exception as e:
            logger.error(f"❌ 数据加载失败：{str(e)}")
            raise
    
    def load_from_db(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        logger.info("从数据库加载数据...")
        
        try:
            from database.schema import PowerWeatherModel
            
            session = db_manager.get_session()
            if session is None:
                logger.warning("⚠ 数据库连接不可用")
                return pd.DataFrame()
            
            query = session.query(PowerWeatherModel)
            
            if limit is not None:
                query = query.limit(limit)
            
            records = query.all()
            
            data_list = [record.to_dict() for record in records]
            self.data = pd.DataFrame(data_list)
            
            logger.info(f"✓ 从数据库加载{len(self.data)}条记录")
            
            session.close()
            
            return self.data
        
        except Exception as e:
            logger.error(f"❌ 从数据库加载失败：{str(e)}")
            return pd.DataFrame()
    
    def get_statistics(self) -> dict:
        """获取数据统计信息"""
        if self.data is None:
            return {}
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': dict(self.data.dtypes),
            'missing_values': dict(self.data.isnull().sum()),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }


class DataExporter:
    """数据导出器 - 导出处理后的数据为标准格式"""
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, output_path: str = 'data/processed.csv',
                     index: bool = False) -> str:
        """导出为CSV"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(path, index=index)
        logger.info(f"✓ 数据已导出到CSV：{path}")
        
        return str(path)
    
    @staticmethod
    def export_to_parquet(data: pd.DataFrame, output_path: str = 'data/processed.parquet') -> str:
        """导出为Parquet"""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data.to_parquet(path, index=False)
            logger.info(f"✓ 数据已导出到Parquet：{path}")
            
            return str(path)
        except ImportError:
            logger.warning("⚠ Parquet库未安装，使用CSV替代")
            return DataExporter.export_to_csv(data)
    
    @staticmethod
    def export_to_numpy(X: np.ndarray, y: np.ndarray, 
                       output_dir: str = 'data/numpy') -> Dict[str, str]:
        """导出为Numpy格式"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        X_path = path / 'X.npy'
        y_path = path / 'y.npy'
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        logger.info(f"✓ Numpy数据已导出到：{output_dir}")
        
        return {
            'X': str(X_path),
            'y': str(y_path)
        }


class TrainingDataPipeline:
    """
    完整的训练数据生成管道
    
    自动处理：
    1. 数据加载
    2. 数据清洗
    3. 特征工程
    4. 数据划分
    5. 导出为标准格式
    """
    
    def __init__(self, csv_path: str = 'data/data.csv', 
                 lookback: int = 24, test_size: float = 0.2):
        """
        初始化训练数据管道
        
        Args:
            csv_path: CSV文件路径
            lookback: 序列长度
            test_size: 测试集比例
        """
        self.csv_path = csv_path
        self.lookback = lookback
        self.test_size = test_size
        
        self.raw_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.training_data = None
        
        logger.info("训练数据管道初始化")
    
    def run(self, nrows: Optional[int] = None, 
            target_col: str = 'actual_power',
            feature_cols: Optional[list] = None,
            output_dir: str = 'data/training',
            batch_size: int = 32) -> Dict[str, Any]:
        """
        运行完整的训练数据生成管道
        
        Args:
            nrows: 最多加载行数
            target_col: 目标列名
            feature_cols: 特征列名列表
            output_dir: 输出目录
            batch_size: 批大小
        
        Returns:
            包含训练数据和元数据的字典
        """
        logger.info("="*60)
        logger.info("运行训练数据管道")
        logger.info("="*60)
        
        try:
            # 第1步：加载数据
            logger.info("\n[1/6] 加载数据...")
            loader = DataLoader(self.csv_path)
            self.raw_data = loader.load(nrows=nrows)
            
            # 第2步：验证数据
            logger.info("\n[2/6] 验证数据...")
            valid, errors = DataSchema.validate(self.raw_data)
            if not valid:
                logger.error(f"数据验证失败：{errors}")
                raise ValueError("数据验证失败")
            
            # 第3步：清洗数据
            logger.info("\n[3/6] 清洗数据...")
            etl = ETLPipeline(self.csv_path)
            self.cleaned_data = etl._clean_data(self.raw_data)
            
            # 第4步：特征工程
            logger.info("\n[4/6] 特征工程...")
            engineer = FeatureEngineer(lookback=self.lookback, test_size=self.test_size)
            
            # 确定特征列
            if feature_cols is None:
                numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != target_col]
            
            # 添加特征
            processed = engineer.add_lag_features(self.cleaned_data, columns=feature_cols, lags=self.lookback)
            processed = engineer.add_rolling_features(processed, columns=feature_cols, windows=[7, 24])
            processed = engineer.normalize(processed, method='standard')
            
            self.processed_data = processed
            
            # 第5步：数据划分和序列化
            logger.info("\n[5/6] 数据划分...")
            
            # 准备特征和标签
            numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
            X = processed[numeric_cols].values
            
            if target_col in numeric_cols:
                y = processed[target_col].values
            else:
                logger.warning(f"⚠ 目标列 {target_col} 不存在，使用第一个数值列")
                y = processed[numeric_cols[0]].values
            
            # 创建序列
            X_seq, y_seq = engineer.create_sequences(X, y, self.lookback)
            
            # 划分数据集
            split_result = engineer.split_data(X_seq, y_seq, self.test_size)
            
            # 第6步：导出数据
            logger.info("\n[6/6] 导出数据...")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 导出训练数据
            result = engineer.export_training_data(
                split_result['X_train'],
                split_result['y_train'],
                split_result['X_test'],
                split_result['y_test'],
                output_dir=output_dir,
                batch_size=batch_size
            )
            
            # 生成总结报告
            report = {
                'status': 'success',
                'raw_data_shape': self.raw_data.shape,
                'cleaned_data_shape': self.cleaned_data.shape,
                'processed_data_shape': self.processed_data.shape,
                'training_data': {
                    'X_train_shape': split_result['X_train'].shape,
                    'X_test_shape': split_result['X_test'].shape,
                    'y_train_shape': split_result['y_train'].shape,
                    'y_test_shape': split_result['y_test'].shape,
                },
                'output_directory': str(output_path),
                'files': {
                    'X_train': str(output_path / 'X_train.npy'),
                    'y_train': str(output_path / 'y_train.npy'),
                    'X_test': str(output_path / 'X_test.npy'),
                    'y_test': str(output_path / 'y_test.npy'),
                    'scaler': str(output_path / 'scaler.pkl'),
                    'metadata': str(output_path / 'metadata.json'),
                },
                'loaders': {
                    'train_loader': result['train_loader'],
                    'test_loader': result['test_loader'],
                }
            }
            
            # 保存报告
            with open(output_path / 'pipeline_report.json', 'w') as f:
                report_copy = report.copy()
                report_copy.pop('loaders')  # 移除不可序列化的对象
                json.dump(report_copy, f, indent=2)
            
            logger.info("="*60)
            logger.info("✓ 训练数据管道完成")
            logger.info("="*60)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ 管道执行失败：{str(e)}")
            import traceback
            traceback.print_exc()
            raise
