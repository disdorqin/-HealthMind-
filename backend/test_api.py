from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend.api import create_app


def run_smoke_test() -> None:
    app = create_app()
    client = app.test_client()

    login_resp = client.post(
        "/api/login",
        json={
            "username": "张三",
            "age": 28,
            "gender": "女",
            "email": "test@example.com",
            "address": "上海",
        },
    )
    assert login_resp.status_code == 200
    login_data = login_resp.get_json()
    assert login_data["success"] is True
    assert login_data["data"]["age"] == 28
    assert login_data["data"]["gender"] == "女"

    user_id = login_data["data"]["user_id"]

    record_resp = client.post(
        "/api/record",
        json={
            "user_id": user_id,
            "date": "2026-03-18",
            "carbon_value": 10.5,
            "predicted_value": 11.0,
        },
    )
    assert record_resp.status_code == 200
    assert record_resp.get_json()["success"] is True

    credits_resp = client.post(
        "/api/credits",
        json={"user_id": user_id, "credits": 10, "reason": "低碳一天"},
    )
    assert credits_resp.status_code == 200
    assert credits_resp.get_json()["success"] is True

    history_resp = client.get(f"/api/history/{user_id}?days=30")
    assert history_resp.status_code == 200
    assert history_resp.get_json()["success"] is True

    print("所有测试通过！")


if __name__ == "__main__":
    run_smoke_test()