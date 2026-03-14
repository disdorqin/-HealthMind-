"""特征加工模块"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from src.core.utils.logger import logger


class FeatureProcessor:
    """特征加工器 - 处理特征工程和数据变换"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.scaler = None
        self.feature_names = None
    
    def add_time_features(self) -> FeatureProcessor:
        """添加时间特征"""
        logger.info("添加时间特征")
        
        # 假设有时间列或需要生成
        if 'Timestamp' in self.data.columns:
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data['hour'] = self.data['Timestamp'].dt.hour
            self.data['day'] = self.data['Timestamp'].dt.day
            self.data['month'] = self.data['Timestamp'].dt.month
            self.data['dayofweek'] = self.data['Timestamp'].dt.dayofweek
            logger.info("时间特征添加成功")
        else:
            logger.info("未找到Timestamp列，跳过时间特征")
        
        return self
    
    def add_lag_features(self, columns: list, lags: int = 24) -> FeatureProcessor:
        """
        添加滞后特征
        
        Args:
            columns: 需要添加滞后特征的列名
            lags: 滞后步数，默认24小时
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"添加滞后特征，滞后步数：{lags}")
        
        for col in columns:
            if col in self.data.columns:
                for lag in range(1, lags + 1):
                    self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
        
        # 删除NaN行
        self.data = self.data.dropna()
        logger.info(f"滞后特征添加完成，当前形状：{self.data.shape}")
        
        return self
    
    def add_rolling_features(self, columns: list, windows: list = [7, 24]) -> FeatureProcessor:
        """
        添加滚动窗口特征
        
        Args:
            columns: 需要添加特征的列名
            windows: 窗口大小列表
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"添加滚动窗口特征，窗口：{windows}")
        
        for col in columns:
            if col in self.data.columns:
                for window in windows:
                    self.data[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window=window).mean()
                    self.data[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window=window).std()
        
        self.data = self.data.dropna()
        logger.info(f"滚动窗口特征添加完成，当前形状：{self.data.shape}")
        
        return self
    
    def normalize(self, method: str = 'standard', numeric_cols: list = None) -> FeatureProcessor:
        """
        特征归一化
        
        Args:
            method: 'standard' 或 'minmax'
            numeric_cols: 需要归一化的列，None表示所有数值列
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"特征归一化，方法：{method}")
        
        if numeric_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的方法：{method}")
        
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        self.feature_names = numeric_cols
        
        logger.info(f"归一化完成，处理列数：{len(numeric_cols)}")
        
        return self
    
    def feature_selection_pca(self, n_components: float = 0.95) -> FeatureProcessor:
        """
        PCA特征选择
        
        Args:
            n_components: 保留的方差比例，0-1之间
        
        Returns:
            self，支持链式调用
        """
        logger.info(f"PCA特征选择，保留方差比例：{n_components}")
        
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            pca = PCA(n_components=n_components)
            
            transformed = pca.fit_transform(self.data[numeric_cols])
            
            # 创建新的列名
            new_cols = [f'pca_{i}' for i in range(transformed.shape[1])]
            
            # 保留原有的非数值列
            non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
            
            # 重新组织数据
            pca_data = pd.DataFrame(transformed, columns=new_cols, index=self.data.index)
            self.data = pd.concat([self.data[non_numeric_cols], pca_data], axis=1)
            
            logger.info(f"PCA完成，维度从 {len(numeric_cols)} 降至 {len(new_cols)}")
        
        except Exception as e:
            logger.warning(f"PCA处理失败：{str(e)}，跳过此步骤")
        
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        """获取处理后的数据"""
        logger.info(f"特征加工完成，最终形状：{self.data.shape}")
        return self.data
