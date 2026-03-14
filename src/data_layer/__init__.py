"""数据层 - 负责数据处理、清洗和特征工程

数据流：
CSV文件 → ETL管道 → 数据验证 → 数据清洗 → 
特征工程 → 数据划分 → 导出为训练格式

模块说明：
- data_loader: 数据加载（CSV/DB）
- data_cleaner: 数据清洗（旧版）
- data_processor: 特征加工（旧版）
- data_schema: 数据模式和完整性检查
- feature_engineering: 特征工程管道
- etl_pipeline: 自动化ETL（CSV→MySQL）
- data_exporter: 数据导出和训练数据生成
- db_manager: 数据库管理器
"""

# 旧版导入（向后兼容）
from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .data_processor import FeatureProcessor

# 新版导入
from .data_schema import DataSchema, DataIntegrityChecker
from .feature_engineering import FeatureEngineer
from .etl_pipeline import ETLPipeline
from .data_exporter import DataExporter, TrainingDataPipeline
from .db_manager import DatabaseManager, db_manager

__all__ = [
    # 旧版（向后兼容）
    'DataLoader',
    'DataCleaner',
    'FeatureProcessor',
    
    # 新版
    'DataSchema',
    'DataIntegrityChecker',
    'FeatureEngineer',
    'ETLPipeline',
    'DataExporter',
    'TrainingDataPipeline',
    'DatabaseManager',
    'db_manager',
]
