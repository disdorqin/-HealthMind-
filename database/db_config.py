"""Database configuration module.

This module initializes the SQLAlchemy 2.0 engine and session factory
using environment variables defined in the project-level .env file.

The configuration targets a MySQL database by default but can be
adapted to other backends by changing the connection URL.
"""

from __future__ import annotations

import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


load_dotenv()


def _build_database_url() -> str:
    """Build the SQLAlchemy database URL from environment variables.

    Returns:
        str: A SQLAlchemy-compatible database URL.
    """
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "Zlt20060313#")
    host = os.getenv("DB_HOST", "sql-container")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "fengmang")

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


DATABASE_URL = _build_database_url()

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
)


def get_db_session() -> Generator[Session, None, None]:
    """Provide a SQLAlchemy session for use in request/processing scope.

    This generator is suitable for use in dependency injection patterns.

    Yields:
        Session: A SQLAlchemy session instance.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

