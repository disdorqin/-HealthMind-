"""数据导入模块"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.core.utils.logger import logger


class DataLoader:
    """数据加载器 - 处理数据导入"""
    
    def __init__(self, data_path: str = 'data/data.csv'):
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
    
    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            nrows: 加载行数，None表示全部加载
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            logger.info(f"开始加载数据：{self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
            
            self.data = pd.read_csv(self.data_path, nrows=nrows)
            logger.info(f"数据加载成功，形状：{self.data.shape}")
            
            return self.data
        
        except Exception as e:
            logger.error(f"数据加载失败：{str(e)}")
            raise
    
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
