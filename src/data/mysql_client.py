from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import os

from config import get_database_settings


def _load_secrets_from_toml() -> Dict[str, Any]:
    secrets_file = Path(".streamlit") / "secrets.toml"
    if not secrets_file.exists():
        return {}

    # Python 3.11+
    import tomllib

    with secrets_file.open("rb") as f:
        return tomllib.load(f)


def read_db_config() -> Dict[str, Any]:
    """Read db config from Streamlit secrets first, then fallback to .streamlit/secrets.toml."""
    try:
        import streamlit as st

        db_section = st.secrets.get("database", {})
        if db_section:
            return dict(db_section)
    except Exception:
        pass

    secrets = _load_secrets_from_toml()
    if secrets.get("database"):
        return dict(secrets.get("database", {}))

    return get_database_settings()


@dataclass
class MySQLConnectionInfo:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_secrets(cls) -> "MySQLConnectionInfo":
        cfg = read_db_config()
        return cls(
            host=str(cfg.get("host", os.getenv("DB_HOST", "localhost"))),
            port=int(cfg.get("port", os.getenv("DB_PORT", 3306))),
            user=str(cfg.get("user", os.getenv("DB_USER", "root"))),
            password=str(cfg.get("password", os.getenv("DB_PASSWORD", ""))),
            database=str(cfg.get("database", os.getenv("DB_NAME", "power"))),
        )
