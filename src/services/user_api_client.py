from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error, parse, request

from backend.user_manager import (
    add_carbon_record as local_add_carbon_record,
    add_credits as local_add_credits,
    build_user_id,
    create_user as local_create_user,
    get_user as local_get_user,
    get_user_history as local_get_user_history,
    summarize_user,
)


@dataclass
class UserAPIClient:
    base_url: str = "http://127.0.0.1:5000"
    timeout: int = 8
    fallback_to_local: bool = True

    def _build_url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        safe_path = parse.quote(path, safe="/:")
        url = self.base_url.rstrip("/") + safe_path
        if params:
            url += "?" + parse.urlencode(params)
        return url

    def _request_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._build_url(path, params=params)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                return json.loads(raw)
            except Exception:
                return {"success": False, "message": f"HTTP {exc.code}"}
        except Exception as exc:
            if not self.fallback_to_local:
                return {"success": False, "message": str(exc)}
            return self._local_fallback(method, path, payload=payload, params=params)

    def _local_fallback(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        if path == "/api/login" and method == "POST":
            username = str(payload.get("username", "")).strip()
            if not username:
                return {"success": False, "message": "用户名不能为空"}
            user_id = str(payload.get("user_id") or build_user_id(username))
            user = local_create_user(
                user_id,
                username,
                age=int(payload["age"]) if payload.get("age") not in {None, ""} else None,
                gender=payload.get("gender"),
                email=payload.get("email"),
                address=payload.get("address"),
            )
            return {"success": True, "message": "登录成功", "data": summarize_user(user)}

        if path.startswith("/api/user/") and method == "GET":
            user_id = parse.unquote(path.rsplit("/", 1)[-1])
            user = local_get_user(user_id)
            if not user:
                return {"success": False, "message": "用户不存在"}
            return {"success": True, "message": "获取成功", "data": summarize_user(user)}

        if path.startswith("/api/history/") and method == "GET":
            user_id = parse.unquote(path.rsplit("/", 1)[-1])
            days = int((params or {}).get("days", 30))
            user = local_get_user(user_id)
            if not user:
                return {"success": False, "message": "用户不存在"}
            return {
                "success": True,
                "message": "获取成功",
                "data": {"user_id": user_id, "records": local_get_user_history(user_id, days=days)},
            }

        if path == "/api/record" and method == "POST":
            user_id = str(payload.get("user_id", "")).strip()
            date = str(payload.get("date", "")).strip()
            carbon_value = payload.get("carbon_value")
            predicted_value = payload.get("predicted_value")
            if not user_id or not date or carbon_value is None:
                return {"success": False, "message": "user_id、date、carbon_value 为必填项"}
            ok = local_add_carbon_record(user_id, date, float(carbon_value), predicted_value=float(predicted_value) if predicted_value is not None else None)
            if not ok:
                return {"success": False, "message": "记录保存失败，用户不存在"}
            return {"success": True, "message": "记录保存成功", "data": {"record_id": f"rec_{date.replace('-', '')}_{user_id}"}}

        if path == "/api/credits" and method == "POST":
            user_id = str(payload.get("user_id", "")).strip()
            credits = payload.get("credits")
            reason = str(payload.get("reason", "")).strip()
            if not user_id or credits is None:
                return {"success": False, "message": "user_id、credits 为必填项"}
            ok = local_add_credits(user_id, int(credits), reason=reason)
            if not ok:
                return {"success": False, "message": "积分添加失败，用户不存在"}
            user = local_get_user(user_id)
            return {"success": True, "message": "积分添加成功", "data": {"new_total": int(user.get("total_credits", 0)) if user else 0}}

        return {"success": False, "message": "本地模式未实现该接口"}

    def login(
        self,
        username: str,
        user_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        email: Optional[str] = None,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request_json("POST", "/api/login", {
            "username": username,
            "user_id": user_id,
            "age": age,
            "gender": gender,
            "email": email,
            "address": address,
        })

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/api/user/{user_id}")

    def get_history(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        return self._request_json("GET", f"/api/history/{user_id}", params={"days": days})

    def add_record(self, user_id: str, date: str, carbon_value: float, predicted_value: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"user_id": user_id, "date": date, "carbon_value": carbon_value}
        if predicted_value is not None:
            payload["predicted_value"] = predicted_value
        return self._request_json("POST", "/api/record", payload)

    def add_credits(self, user_id: str, credits: int, reason: str = "") -> Dict[str, Any]:
        return self._request_json("POST", "/api/credits", {"user_id": user_id, "credits": credits, "reason": reason})
