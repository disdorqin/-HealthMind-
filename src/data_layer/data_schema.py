"""数据完整性和模式校验"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.core.utils.logger import logger


class DataSchema:
    """数据模式定义 - 定义必需列和数据类型"""
    
    # 必需的数据列
    REQUIRED_COLUMNS = {
        'timestamp': 'datetime64',  # 时间戳
        'actual_power': 'float64',   # 实际功率
        'wind_speed': 'float64',     # 风速
        'temperature': 'float64',    # 温度
        'irradiance': 'float64',     # 太阳辐照度
    }
    
    # 可选列
    OPTIONAL_COLUMNS = {
        'is_holiday': 'int64',       # 是否假期
        'hour_of_day': 'int64',      # 小时
        'day_of_week': 'int64',      # 周几
    }
    
    # 数据有效范围
    VALUE_RANGES = {
        'actual_power': (0, 1000),   # MW
        'wind_speed': (0, 30),       # m/s
        'temperature': (-50, 50),    # ℃
        'irradiance': (0, 1400),     # W/m²
    }
    
    @classmethod
    def validate(cls, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证数据完整性
        
        Args:
            data: 输入DataFrame
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需列
        missing_cols = [col for col in cls.REQUIRED_COLUMNS.keys() if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少必需列：{', '.join(missing_cols)}")
            logger.warning(f"⚠ 缺少必需列：{missing_cols}")
        
        # 检查行数
        if len(data) == 0:
            errors.append("数据为空（0行）")
            logger.warning("⚠ 数据为空")
        
        # 检查缺失值
        missing_counts = data.isnull().sum()
        high_missing = missing_counts[missing_counts > len(data) * 0.5]
        if len(high_missing) > 0:
            errors.append(f"列缺失值过多（>50%）：{', '.join(high_missing.index)}")
            logger.warning(f"⚠ 这些列缺失值过多：{high_missing.to_dict()}")
        
        # 检查数值范围
        for col, (min_val, max_val) in cls.VALUE_RANGES.items():
            if col in data.columns:
                invalid = data[(data[col] < min_val) | (data[col] > max_val)]
                if len(invalid) > 0:
                    logger.warning(f"⚠ 列 {col} 有 {len(invalid)} 个值超出范围 [{min_val}, {max_val}]")
        
        return len(errors) == 0, errors
    
    @classmethod
    def get_required_columns(cls) -> List[str]:
        """获取必需列列表"""
        return list(cls.REQUIRED_COLUMNS.keys())
    
    @classmethod
    def get_all_columns(cls) -> Dict[str, str]:
        """获取所有列（必需 + 可选）"""
        return {**cls.REQUIRED_COLUMNS, **cls.OPTIONAL_COLUMNS}


class DataIntegrityChecker:
    """数据完整性检查器"""
    
    @staticmethod
    def check_duplicates(data: pd.DataFrame, subset: List[str] = None) -> Dict:
        """检查重复行"""
        if subset is None:
            subset = ['timestamp']
        
        dup_count = data.duplicated(subset=subset).sum()
        
        result = {
            'has_duplicates': dup_count > 0,
            'duplicate_count': int(dup_count),
            'duplicate_ratio': float(dup_count / len(data)) if len(data) > 0 else 0
        }
        
        if result['has_duplicates']:
            logger.warning(f"⚠ 检测到 {dup_count} 行重复数据")
        
        return result
    
    @staticmethod
    def check_missing_values(data: pd.DataFrame) -> Dict:
        """检查缺失值"""
        missing_counts = data.isnull().sum()
        missing_pcts = (missing_counts / len(data) * 100).round(2)
        
        result = {
            'has_missing': missing_counts.sum() > 0,
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_pcts.to_dict()
        }
        
        for col, count in missing_counts[missing_counts > 0].items():
            pct = missing_pcts[col]
            logger.info(f"  列 {col}: {count} 缺失值 ({pct}%)")
        
        return result
    
    @staticmethod
    def check_data_types(data: pd.DataFrame, schema: Dict[str, str]) -> Dict:
        """检查数据类型"""
        mismatches = {}
        
        for col, expected_dtype in schema.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                if actual_dtype != expected_dtype:
                    mismatches[col] = {
                        'expected': expected_dtype,
                        'actual': actual_dtype
                    }
        
        if mismatches:
            logger.warning(f"⚠ 数据类型不匹配：{mismatches}")
        
        return {
            'has_mismatches': len(mismatches) > 0,
            'mismatches': mismatches
        }
    
    @staticmethod
    def generate_report(data: pd.DataFrame) -> str:
        """生成完整的数据完整性报告"""
        valid, errors = DataSchema.validate(data)
        
        dup_check = DataIntegrityChecker.check_duplicates(data)
        missing_check = DataIntegrityChecker.check_missing_values(data)
        
        report = []
        report.append("=" * 60)
        report.append("📊 数据完整性检查报告")
        report.append("=" * 60)
        report.append(f"\n总行数: {len(data)}")
        report.append(f"总列数: {len(data.columns)}")
        
        if not valid:
            report.append("\n❌ 验证失败:")
            for error in errors:
                report.append(f"  - {error}")
        else:
            report.append("\n✓ 基本验证通过")
        
        if dup_check['has_duplicates']:
            report.append(f"\n⚠ 重复行检查:")
            report.append(f"  - 数量: {dup_check['duplicate_count']}")
            report.append(f"  - 比率: {dup_check['duplicate_ratio']:.2%}")
        
        if missing_check['has_missing']:
            report.append("\n⚠ 缺失值检查:")
            for col, count in missing_check['missing_counts'].items():
                if count > 0:
                    pct = missing_check['missing_percentages'][col]
                    report.append(f"  - {col}: {count} ({pct}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
