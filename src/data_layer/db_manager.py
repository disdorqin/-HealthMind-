"""数据库配置和ETL集成"""

from __future__ import annotations

import os
from typing import Generator, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from src.core.utils.logger import logger

load_dotenv()


def _build_database_url() -> str:
    """构建SQLAlchemy数据库URL"""
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "power_trading")
    
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


class DatabaseManager:
    """数据库管理器 - 统一的数据库操作接口"""
    
    _instance = None
    _engine = None
    _SessionLocal = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine()
        return cls._instance
    
    def _init_engine(self):
        """初始化数据库引擎"""
        try:
            DATABASE_URL = _build_database_url()
            DatabaseManager._engine = create_engine(
                DATABASE_URL,
                echo=False,
                pool_pre_ping=True,
                future=True,
                connect_args={"charset": "utf8mb4"}
            )
            DatabaseManager._SessionLocal = sessionmaker(
                bind=DatabaseManager._engine,
                autocommit=False,
                autoflush=False,
            )
            logger.info("✓ 数据库引擎初始化成功")
        except Exception as e:
            logger.error(f"数据库引擎初始化失败：{str(e)}")
            DatabaseManager._engine = None
            DatabaseManager._SessionLocal = None
    
    @classmethod
    def get_engine(cls):
        """获取数据库引擎"""
        if cls._engine is None:
            instance = cls()
        return cls._engine
    
    @classmethod
    def get_session(cls) -> Optional[Session]:
        """获取数据库会话"""
        if cls._SessionLocal is None:
            instance = cls()
        
        if cls._SessionLocal is None:
            return None
        
        return cls._SessionLocal()
    
    @classmethod
    def create_tables(cls, base):
        """创建所有表"""
        engine = cls.get_engine()
        if engine is None:
            logger.warning("⚠ 数据库连接不可用，跳过建表")
            return
        
        try:
            base.metadata.create_all(bind=engine)
            logger.info("✓ 数据库表创建成功")
        except Exception as e:
            logger.error(f"创建表失败：{str(e)}")
    
    @classmethod
    def table_exists(cls, table_name: str) -> bool:
        """检查表是否存在"""
        engine = cls.get_engine()
        if engine is None:
            return False
        
        try:
            inspector = inspect(engine)
            return table_name in inspector.get_table_names()
        except Exception as e:
            logger.error(f"检查表失败：{str(e)}")
            return False


# 全局数据库管理器实例
db_manager = DatabaseManager()
