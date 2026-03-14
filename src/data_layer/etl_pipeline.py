"""自动化ETL管道 - 从CSV增量导入MySQL"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib

import pandas as pd
import numpy as np
from sqlalchemy import select, func

from src.core.utils.logger import logger
from .data_schema import DataSchema, DataIntegrityChecker
from .feature_engineering import FeatureEngineer
from .db_manager import db_manager


class ETLPipeline:
    """
    自动化ETL管道
    
    支持从CSV文件增量导入到MySQL数据库，并自动进行数据清洗和特征工程
    """
    
    def __init__(self, csv_path: str = 'data/data.csv', 
                 table_name: str = 'power_weather'):
        """
        初始化ETL管道
        
        Args:
            csv_path: CSV文件路径
            table_name: 目标数据库表名
        """
        self.csv_path = Path(csv_path)
        self.table_name = table_name
        
        # 状态追踪文件
        self.state_file = Path('data/.etl_state.json')
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.last_processed_hash = None
        self.new_records_count = 0
        
        logger.info(f"ETL管道初始化：CSV={csv_path}, 表={table_name}")
    
    def _load_csv(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载CSV文件"""
        if not self.csv_path.exists():
            logger.error(f"❌ CSV文件不存在：{self.csv_path}")
            raise FileNotFoundError(f"数据文件不存在：{self.csv_path}")
        
        logger.info(f"加载CSV文件：{self.csv_path}")
        
        try:
            data = pd.read_csv(self.csv_path, nrows=nrows)
            logger.info(f"✓ CSV加载成功，形状：{data.shape}")
            return data
        except Exception as e:
            logger.error(f"❌ CSV加载失败：{str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据完整性"""
        logger.info("验证数据完整性...")
        
        # 使用DataSchema进行验证
        valid, errors = DataSchema.validate(data)
        
        if not valid:
            logger.warning(f"⚠ 数据验证失败：{errors}")
            
            # 自动跳过缺失关键列的数据
            missing_cols = [col for col in DataSchema.REQUIRED_COLUMNS.keys() 
                          if col not in data.columns]
            
            if missing_cols:
                logger.error(f"❌ 缺少关键列，无法继续处理：{missing_cols}")
                raise ValueError(f"缺少关键列：{missing_cols}")
        
        # 生成完整性报告
        report = DataIntegrityChecker.generate_report(data)
        logger.info(report)
        
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        logger.info("清洗数据...")
        
        # 删除重复行
        dup_before = len(data)
        data = data.drop_duplicates(subset=['timestamp'])
        dup_removed = dup_before - len(data)
        
        if dup_removed > 0:
            logger.info(f"  - 删除{dup_removed}行重复数据")
        
        # 处理缺失值（forward fill）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_before = data[col].isnull().sum()
            if missing_before > 0:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                logger.info(f"  - 修复列 {col} 的{missing_before}个缺失值")
        
        # 处理异常值（IQR）
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            
            outliers = (data[col] < lower) | (data[col] > upper)
            if outliers.sum() > 0:
                data.loc[data[col] < lower, col] = lower
                data.loc[data[col] > upper, col] = upper
                logger.info(f"  - 修复列 {col} 的{outliers.sum()}个异常值")
        
        logger.info(f"✓ 数据清洗完成，最终形状：{data.shape}")
        
        return data
    
    def _detect_new_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """检测并返回新记录"""
        logger.info("检测新记录...")
        
        # 计算数据hash用于增量检测
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()
        
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    last_hash = state.get('last_hash')
                    
                    if last_hash == data_hash:
                        logger.info("✓ 没有新记录")
                        return pd.DataFrame()
            except Exception as e:
                logger.warning(f"⚠ 读取状态文件失败：{str(e)}")
        
        # 保存新的状态
        try:
            import json
            with open(self.state_file, 'w') as f:
                json.dump({'last_hash': data_hash, 'timestamp': datetime.now().isoformat()}, f)
        except Exception as e:
            logger.warning(f"⚠ 保存状态文件失败：{str(e)}")
        
        self.new_records_count = len(data)
        logger.info(f"✓ 检测到{self.new_records_count}条新记录")
        
        return data
    
    def _insert_to_db(self, data: pd.DataFrame) -> int:
        """插入数据到MySQL"""
        logger.info(f"插入数据到数据库表 {self.table_name}...")
        
        session = db_manager.get_session()
        if session is None:
            logger.warning("⚠ 数据库连接不可用，跳过数据库插入")
            return 0
        
        try:
            from database.schema import PowerWeatherModel
            
            inserted_count = 0
            
            for idx, row in data.iterrows():
                # 检查记录是否已存在
                existing = session.query(PowerWeatherModel).filter_by(
                    timestamp=row.get('timestamp')
                ).first()
                
                if existing is None:
                    # 创建新记录
                    record = PowerWeatherModel(
                        timestamp=row.get('timestamp'),
                        actual_power=float(row.get('actual_power')) if pd.notna(row.get('actual_power')) else None,
                        wind_speed=float(row.get('wind_speed')) if pd.notna(row.get('wind_speed')) else None,
                        temperature=float(row.get('temperature')) if pd.notna(row.get('temperature')) else None,
                        irradiance=float(row.get('irradiance')) if pd.notna(row.get('irradiance')) else None,
                        is_holiday=int(row.get('is_holiday', 0)),
                        hour_of_day=int(row.get('hour_of_day', 0))
                    )
                    session.add(record)
                    inserted_count += 1
            
            session.commit()
            logger.info(f"✓ 成功插入{inserted_count}条新记录")
            
            return inserted_count
        
        except Exception as e:
            logger.error(f"❌ 数据库插入失败：{str(e)}")
            session.rollback()
            return 0
        
        finally:
            session.close()
    
    def run_etl(self, nrows: Optional[int] = None) -> Dict:
        """
        运行完整的ETL流程
        
        Args:
            nrows: 最多加载行数
        
        Returns:
            ETL执行结果
        """
        logger.info("="*60)
        logger.info("开始ETL流程")
        logger.info("="*60)
        
        result = {
            'success': False,
            'csv_rows': 0,
            'validated_rows': 0,
            'cleaned_rows': 0,
            'new_records': 0,
            'inserted_rows': 0,
            'errors': []
        }
        
        try:
            # 第1步：加载CSV
            data = self._load_csv(nrows=nrows)
            result['csv_rows'] = len(data)
            
            # 第2步：验证数据
            data = self._validate_data(data)
            result['validated_rows'] = len(data)
            
            # 第3步：清洗数据
            data = self._clean_data(data)
            result['cleaned_rows'] = len(data)
            
            # 第4步：检测新记录
            new_data = self._detect_new_records(data)
            result['new_records'] = len(new_data)
            
            # 第5步：插入数据库
            if len(new_data) > 0:
                inserted = self._insert_to_db(new_data)
                result['inserted_rows'] = inserted
            
            result['success'] = True
            
            logger.info("="*60)
            logger.info("✓ ETL流程完成")
            logger.info("="*60)
            logger.info(f"CSV行数: {result['csv_rows']}")
            logger.info(f"验证行数: {result['validated_rows']}")
            logger.info(f"清洗行数: {result['cleaned_rows']}")
            logger.info(f"新记录数: {result['new_records']}")
            logger.info(f"插入行数: {result['inserted_rows']}")
            logger.info("="*60)
        
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"❌ ETL流程失败：{str(e)}")
        
        return result
