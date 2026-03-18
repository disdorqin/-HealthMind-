from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path(__file__).resolve().parent / "user_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify_username(username: str) -> str:
    normalized = re.sub(r"\s+", "", username.strip().lower())
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", normalized)
    if normalized:
        return normalized[:24]
    digest = hashlib.sha1(username.encode("utf-8")).hexdigest()[:10]
    return f"user_{digest}"


def build_user_id(username: str) -> str:
    """Create a stable user id from the username."""
    return _slugify_username(username)


def get_user_file(user_id: str) -> Path:
    return DATA_DIR / f"{user_id}.json"


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_user(
    user_id: str,
    username: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "username": username,
        "created_at": _now(),
        "total_credits": 0,
        "carbon_records": [],
        "credit_history": [],
        "profile": {
            "age": age,
            "gender": gender,
            "email": email,
            "address": address,
        },
        "settings": {
            "weekly_budget": 350,
            "diet_type": "omnivore",
            "notifications_enabled": True,
        },
    }


def _read_user(user_id: str) -> Optional[Dict[str, Any]]:
    user_file = get_user_file(user_id)
    if not user_file.exists():
        return None
    with user_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_user(user_id: str, user_data: Dict[str, Any]) -> None:
    user_file = get_user_file(user_id)
    tmp_file = user_file.with_suffix(".json.tmp")
    with tmp_file.open("w", encoding="utf-8") as handle:
        json.dump(user_data, handle, ensure_ascii=False, indent=2)
    tmp_file.replace(user_file)


def create_user(
    user_id: str,
    username: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a user or return the existing one."""
    with _LOCK:
        existing = _read_user(user_id)
        if existing:
            profile = existing.setdefault("profile", {})
            if age is not None and profile.get("age") is None:
                profile["age"] = int(age)
            if gender and not profile.get("gender"):
                profile["gender"] = gender
            if email and not profile.get("email"):
                profile["email"] = email
            if address and not profile.get("address"):
                profile["address"] = address
            _write_user(user_id, existing)
            return existing

        user_data = _default_user(
            user_id=user_id,
            username=username,
            age=int(age) if age is not None else None,
            gender=_normalize_optional_text(gender),
            email=_normalize_optional_text(email),
            address=_normalize_optional_text(address),
        )
        _write_user(user_id, user_data)
        return user_data


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _read_user(user_id)


def add_carbon_record(
    user_id: str,
    date: str,
    carbon_value: float,
    predicted_value: Optional[float] = None,
) -> bool:
    """Append a carbon record to the user's history."""
    with _LOCK:
        user_data = _read_user(user_id)
        if not user_data:
            return False

        record = {
            "record_id": f"rec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "date": date,
            "carbon_value": float(carbon_value),
            "predicted_value": float(predicted_value) if predicted_value is not None else float(carbon_value),
            "timestamp": _now(),
        }
        user_data.setdefault("carbon_records", []).append(record)
        _write_user(user_id, user_data)
        return True


def get_user_history(user_id: str, days: int = 30) -> List[Dict[str, Any]]:
    with _LOCK:
        user_data = _read_user(user_id)
        if not user_data:
            return []

        records = user_data.get("carbon_records", [])
        return records[-max(int(days), 1):]


def add_credits(user_id: str, credits: int, reason: str = "") -> bool:
    with _LOCK:
        user_data = _read_user(user_id)
        if not user_data:
            return False

        credits = int(credits)
        user_data["total_credits"] = int(user_data.get("total_credits", 0)) + credits
        user_data.setdefault("credit_history", []).append(
            {
                "credits": credits,
                "reason": reason,
                "timestamp": _now(),
            }
        )
        _write_user(user_id, user_data)
        return True


def summarize_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the API-friendly user summary."""
    profile = user_data.get("profile", {}) or {}
    return {
        "user_id": user_data.get("user_id"),
        "username": user_data.get("username"),
        "created_at": user_data.get("created_at"),
        "total_credits": user_data.get("total_credits", 0),
        "record_count": len(user_data.get("carbon_records", [])),
        "age": profile.get("age"),
        "gender": profile.get("gender"),
        "email": profile.get("email"),
        "address": profile.get("address"),
        "profile_completed": any(profile.get(key) for key in ["age", "gender", "email", "address"]),
    }
