"""数据清洗模块"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from src.core.utils.logger import logger


class DataCleaner:
    """数据清洗器 - 处理缺失值、异常值等"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_shape = data.shape
    
    def handle_missing_values(self, method: str = 'forward_fill') -> DataCleaner:
        """
        处理缺失值
        
        Args:
            method: 处理方法，'forward_fill' 或 'mean'
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"处理缺失值，方法：{method}")
        missing_count = self.data.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("数据中没有缺失值")
            return self
        
        if method == 'forward_fill':
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            # 只对数值列进行填充
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        else:
            raise ValueError(f"不支持的方法：{method}")
        
        logger.info(f"缺失值处理完成，处理数量：{missing_count}")
        return self
    
    def remove_duplicates(self) -> DataCleaner:
        """删除重复行"""
        logger.info("删除重复行")
        dup_count = self.data.duplicated().sum()
        
        if dup_count > 0:
            self.data = self.data.drop_duplicates()
            logger.info(f"删除重复行数：{dup_count}")
        else:
            logger.info("数据中没有重复行")
        
        return self
    
    def handle_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> DataCleaner:
        """
        处理异常值
        
        Args:
            method: 'iqr' 或 'zscore'
            threshold: IQR方法中的倍数，zscore方法中的标准差倍数
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"处理异常值，方法：{method}")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers = (self.data[col] < lower) | (self.data[col] > upper)
                if outliers.sum() > 0:
                    logger.info(f"列 {col} 检测到 {outliers.sum()} 个异常值")
                    # 用边界值替换异常值
                    self.data.loc[self.data[col] < lower, col] = lower
                    self.data.loc[self.data[col] > upper, col] = upper
        
        elif method == 'zscore':
            try:
                from scipy import stats
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                    threshold_val = threshold
                    outliers = z_scores > threshold_val
                    if outliers.sum() > 0:
                        logger.info(f"列 {col} 检测到 {outliers.sum()} 个异常值")
            except ImportError:
                logger.warning("scipy未安装，使用IQR方法替代")
                return self.handle_outliers(method='iqr', threshold=threshold)
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """获取清洗后的数据"""
        logger.info(f"清洗完成，原始形状：{self.original_shape}，清洗后形状：{self.data.shape}")
        return self.data
