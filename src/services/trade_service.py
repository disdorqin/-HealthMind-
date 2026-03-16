from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from config import get_trade_settings


@dataclass
class TradeAdvice:
    horizon: str
    message: str
    level: str
    expected_revenue: float
    risk_score: float
    actions: List[str]


class HorizonTradeAdvisor:
    """Generate trade advice for day/week/month horizons."""

    def __init__(self, peak_price: float = 1.2, flat_price: float = 0.8, valley_price: float = 0.4):
        cfg = get_trade_settings()
        self.peak_price = float(cfg.get("peak_price", peak_price))
        self.flat_price = float(cfg.get("flat_price", flat_price))
        self.valley_price = float(cfg.get("valley_price", valley_price))
        self.sell_signal_threshold_pct = float(cfg.get("sell_signal_threshold_pct", 3.0))
        self.buy_signal_threshold_pct = float(cfg.get("buy_signal_threshold_pct", -2.0))

    def _hour_price(self, hour: int) -> float:
        if 11 <= hour <= 14 or 18 <= hour <= 23:
            return self.peak_price
        if 23 <= hour or hour < 7:
            return self.valley_price
        return self.flat_price

    @staticmethod
    def _risk_score(values: np.ndarray) -> float:
        mean_val = float(np.mean(values)) if values.size else 0.0
        std_val = float(np.std(values)) if values.size else 0.0
        if mean_val <= 1e-8:
            return 0.0
        score = min(10.0, (std_val / mean_val) * 18.0)
        return float(score)

    def _daily_advice(self, series_96: np.ndarray) -> TradeAdvice:
        hourly_24 = series_96.reshape(24, 4).mean(axis=1) if series_96.size >= 96 else np.resize(series_96, 24)
        prices = np.array([self._hour_price(h) for h in range(24)], dtype=np.float32)
        low_mask = prices <= self.flat_price
        high_mask = prices >= self.peak_price
        buy_cost = float(np.sum(hourly_24[low_mask] * prices[low_mask]) * 0.95)
        sell_revenue = float(np.sum(hourly_24[high_mask] * prices[high_mask]) * 0.85)

        expected_revenue = max(0.0, sell_revenue - buy_cost)
        risk_score = self._risk_score(hourly_24)

        base = float(np.mean(hourly_24)) if hourly_24.size else 0.0
        delta_pct = 0.0
        if hourly_24.size >= 2 and abs(base) > 1e-8:
            delta_pct = float((hourly_24[-1] - hourly_24[0]) / base * 100.0)

        level = "success" if expected_revenue > 0 and risk_score < 6 else "warning"
        if delta_pct >= self.sell_signal_threshold_pct:
            message = "下一天预计价格上行，建议提高峰段卖电比例。"
        elif delta_pct <= self.buy_signal_threshold_pct:
            message = "下一天预计价格走弱，建议增加低谷充电并谨慎卖电。"
        else:
            message = "下一天建议执行谷充峰放，关注晚高峰放电窗口。"
        actions = [
            "00:00-07:00 低谷充电，保证储能SOC>80%",
            "11:00-14:00 与 18:00-23:00 优先放电",
            "实时跟踪负荷突增，保留10%-15%应急容量",
        ]

        return TradeAdvice(
            horizon="day",
            message=message,
            level=level,
            expected_revenue=expected_revenue,
            risk_score=risk_score,
            actions=actions,
        )

    def _weekly_advice(self, series_168: np.ndarray) -> TradeAdvice:
        base = float(np.mean(series_168)) if series_168.size else 0.0
        spread = float(np.percentile(series_168, 85) - np.percentile(series_168, 15)) if series_168.size else 0.0
        expected_revenue = max(0.0, spread * 0.45)
        risk_score = self._risk_score(series_168)

        level = "success" if risk_score < 5 else "warning"
        message = "下周建议加大储能准备，天气驱动波动增强时优先保障容量。"
        actions = [
            "按天气预报滚动更新次日充放计划",
            "周内高负荷日提前12小时完成储能预热",
            "对高温/寒潮日设置更高备用系数",
        ]

        return TradeAdvice(
            horizon="week",
            message=message,
            level=level,
            expected_revenue=expected_revenue,
            risk_score=risk_score,
            actions=actions,
        )

    def _monthly_advice(self, series_30: np.ndarray) -> TradeAdvice:
        trend = float(series_30[-1] - series_30[0]) if series_30.size >= 2 else 0.0
        expected_revenue = max(0.0, abs(trend) * 3.2)
        risk_score = self._risk_score(series_30)

        level = "warning" if risk_score >= 6 else "success"
        message = "月度建议以趋势策略为主，结合合约电与检修计划优化。"
        actions = [
            "按周滚动校正月度采购计划",
            "高趋势上行月份提前锁定部分电量合约",
            "安排检修避开预测高负荷日",
        ]

        return TradeAdvice(
            horizon="month",
            message=message,
            level=level,
            expected_revenue=expected_revenue,
            risk_score=risk_score,
            actions=actions,
        )

    def advise(self, horizon: str, predictions: np.ndarray) -> Dict[str, Any]:
        horizon = horizon.lower().strip()
        if horizon == "day":
            payload = self._daily_advice(predictions)
        elif horizon == "week":
            payload = self._weekly_advice(predictions)
        elif horizon == "month":
            payload = self._monthly_advice(predictions)
        else:
            raise ValueError(f"Unsupported horizon: {horizon}")

        return {
            "horizon": payload.horizon,
            "message": payload.message,
            "level": payload.level,
            "expected_revenue": payload.expected_revenue,
            "risk_score": payload.risk_score,
            "actions": payload.actions,
        }
