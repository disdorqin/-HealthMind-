from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from backend.user_manager import (  # type: ignore
        add_carbon_record,
        add_credits,
        build_user_id,
        create_user,
        get_user,
        get_user_history,
        summarize_user,
    )
else:
    from .user_manager import (
        add_carbon_record,
        add_credits,
        build_user_id,
        create_user,
        get_user,
        get_user_history,
        summarize_user,
    )


def _json_response(success: bool, message: str, data: Dict[str, Any] | None = None, status_code: int = 200):
    payload: Dict[str, Any] = {"success": success, "message": message}
    if data is not None:
        payload["data"] = data
    return jsonify(payload), status_code


def create_app() -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return _json_response(True, "服务正常", {"status": "ok"})

    @app.route("/api/login", methods=["POST", "OPTIONS"])
    def login():
        if request.method == "OPTIONS":
            return _json_response(True, "ok")

        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        if not username:
            return _json_response(False, "用户名不能为空", status_code=400)

        user_id = str(payload.get("user_id") or build_user_id(username))
        age = payload.get("age")
        gender = payload.get("gender")
        email = payload.get("email")
        address = payload.get("address")
        user = create_user(
            user_id,
            username,
            age=int(age) if age not in {None, ""} else None,
            gender=gender,
            email=email,
            address=address,
        )
        return _json_response(True, "登录成功", summarize_user(user))

    @app.route("/api/user/<user_id>", methods=["GET"])
    def user_profile(user_id: str):
        user = get_user(user_id)
        if not user:
            return _json_response(False, "用户不存在", status_code=404)
        return _json_response(True, "获取成功", summarize_user(user))

    @app.route("/api/history/<user_id>", methods=["GET"])
    def history(user_id: str):
        days = int(request.args.get("days", 30))
        user = get_user(user_id)
        if not user:
            return _json_response(False, "用户不存在", status_code=404)
        return _json_response(
            True,
            "获取成功",
            {
                "user_id": user_id,
                "records": get_user_history(user_id, days=days),
            },
        )

    @app.route("/api/record", methods=["POST", "OPTIONS"])
    def record():
        if request.method == "OPTIONS":
            return _json_response(True, "ok")

        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id", "")).strip()
        date = str(payload.get("date", "")).strip()
        carbon_value = payload.get("carbon_value")
        predicted_value = payload.get("predicted_value")

        if not user_id or not date or carbon_value is None:
            return _json_response(False, "user_id、date、carbon_value 为必填项", status_code=400)

        if not add_carbon_record(user_id, date, float(carbon_value), predicted_value=float(predicted_value) if predicted_value is not None else None):
            return _json_response(False, "记录保存失败，用户不存在", status_code=404)

        record_id = f"rec_{date.replace('-', '')}_{user_id}"
        return _json_response(True, "记录保存成功", {"record_id": record_id})

    @app.route("/api/credits", methods=["POST", "OPTIONS"])
    def credits():
        if request.method == "OPTIONS":
            return _json_response(True, "ok")

        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id", "")).strip()
        credits_value = payload.get("credits")
        reason = str(payload.get("reason", "")).strip()

        if not user_id or credits_value is None:
            return _json_response(False, "user_id、credits 为必填项", status_code=400)

        if not add_credits(user_id, int(credits_value), reason=reason):
            return _json_response(False, "积分添加失败，用户不存在", status_code=404)

        user = get_user(user_id)
        return _json_response(True, "积分添加成功", {"new_total": int(user.get("total_credits", 0)) if user else 0})

    @app.errorhandler(404)
    def not_found(_error):
        return _json_response(False, "接口不存在", status_code=404)

    @app.errorhandler(500)
    def internal_error(_error):
        return _json_response(False, "服务器内部错误", status_code=500)

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    app.run(host=host, port=port, debug=debug)