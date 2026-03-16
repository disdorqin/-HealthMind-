from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    settings_path = _root_dir() / "config" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError("settings.yaml root must be a mapping")

    return payload


def get_path(name: str, default: str | None = None) -> str:
    paths = load_settings().get("paths", {})
    if name in paths:
        return str(paths[name])
    if default is not None:
        return default
    raise KeyError(f"Path '{name}' not found in settings")


def get_training_defaults() -> Dict[str, Any]:
    return dict(load_settings().get("training", {}))


def get_model_params(model_name: str) -> Dict[str, Any]:
    models = load_settings().get("models", {})
    return dict(models.get(model_name, {}))


def get_horizon_settings() -> Dict[str, Any]:
    return dict(load_settings().get("horizon", {}))


def get_trade_settings() -> Dict[str, Any]:
    business = load_settings().get("business", {})
    return dict(business.get("trade", {}))


def get_frontend_settings() -> Dict[str, Any]:
    return dict(load_settings().get("frontend", {}))


def get_database_settings() -> Dict[str, Any]:
    return dict(load_settings().get("database", {}))


def clear_settings_cache() -> None:
    load_settings.cache_clear()
