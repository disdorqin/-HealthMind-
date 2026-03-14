"""交易优化器 - 基于预测功率和分时电价的交易建议"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class TradeAdvice:
    """交易建议"""
    action: str  # "buy" or "sell"
    hour: int
    power: float
    price: float
    revenue: float
    reason: str


@dataclass
class TradeOptimizationResult:
    """交易优化结果"""
    buy_advice: List[TradeAdvice]
    sell_advice: List[TradeAdvice]
    expected_revenue: float
    cost_saving: float
    peak_shaving_power: float
    valley_filling_power: float
    summary: Dict[str, Any]


class TradeOptimizer:
    """交易优化器"""
    
    # 分时电价（元/kWh）
    TIME_OF_USE_PRICES = {
        'peak': 0.704,    # 高峰时段
        'flat': 0.604,    # 平段
        'valley': 0.504,  # 低谷时段
    }
    
    # 时段划分（小时）
    PEAK_HOURS = list(range(11, 14)) + list(range(18, 23))  # 11-14 点，18-23 点
    VALLEY_HOURS = list(range(23, 24)) + list(range(0, 7))   # 23-7 点
    
    def __init__(self):
        self.prices = self._build_hourly_prices()
        self.flat_hours = [h for h in range(24) if h not in self.PEAK_HOURS and h not in self.VALLEY_HOURS]
    
    def _build_hourly_prices(self) -> List[float]:
        """构建 24 小时电价列表"""
        prices = []
        for hour in range(24):
            if hour in self.PEAK_HOURS:
                prices.append(self.TIME_OF_USE_PRICES['peak'])
            elif hour in self.VALLEY_HOURS:
                prices.append(self.TIME_OF_USE_PRICES['valley'])
            else:
                prices.append(self.TIME_OF_USE_PRICES['flat'])
        return prices
    
    def _get_price_type(self, hour: int) -> str:
        """获取时段的电价类型"""
        if hour in self.PEAK_HOURS:
            return 'peak'
        elif hour in self.VALLEY_HOURS:
            return 'valley'
        return 'flat'
    
    def optimize(self, predicted_power: np.ndarray, 
                 actual_power: np.ndarray = None) -> TradeOptimizationResult:
        """
        优化交易策略
        
        Args:
            predicted_power: 预测功率数组（24 小时或 96 点）
            actual_power: 实际功率数组（可选，用于对比）
        
        Returns:
            TradeOptimizationResult: 优化结果
        """
        # 确保是 24 小时数据
        if len(predicted_power) > 24:
            # 如果是 96 点（15 分钟粒度），转换为小时
            predicted_hourly = np.array([
                np.mean(predicted_power[i*4:(i+1)*4]) for i in range(24)
            ])
        else:
            predicted_hourly = predicted_power[:24]
        
        avg_power = np.mean(predicted_hourly)
        
        # 生成买卖建议
        buy_advice = []
        sell_advice = []
        
        for hour in range(24):
            power = float(predicted_hourly[hour])
            price = self.prices[hour]
            price_type = self._get_price_type(hour)
            
            if price_type == 'valley':
                # 低谷时段：建议买电（充电/储能）
                reason = f"低谷电价 ({price:.3f}元/kWh)，适合充电储能"
                if power < avg_power * 0.8:
                    reason += f"，预测功率较低 ({power:.2f}kW)"
                buy_advice.append(TradeAdvice(
                    action='buy',
                    hour=hour,
                    power=power,
                    price=price,
                    revenue=power * price,
                    reason=reason
                ))
            elif price_type == 'peak':
                # 高峰时段：建议卖电（放电/自用）
                reason = f"高峰电价 ({price:.3f}元/kWh)，适合放电自用"
                if power > avg_power * 1.2:
                    reason += f"，预测功率较高 ({power:.2f}kW)"
                sell_advice.append(TradeAdvice(
                    action='sell',
                    hour=hour,
                    power=power,
                    price=price,
                    revenue=power * price,
                    reason=reason
                ))
        
        # 计算优化收益
        buy_revenue = sum(a.revenue for a in buy_advice)
        sell_revenue = sum(a.revenue for a in sell_advice)
        expected_revenue = sell_revenue - buy_revenue
        
        # 计算成本节约（相比平均电价）
        avg_price = np.mean(self.prices)
        baseline_cost = np.sum(predicted_hourly) * avg_price
        optimized_cost = buy_revenue
        cost_saving = baseline_cost - optimized_cost if baseline_cost > 0 else 0
        
        # 削峰填谷计算
        peak_power = np.max(predicted_hourly)
        valley_power = np.min(predicted_hourly)
        peak_shaving_power = max(0, peak_power - avg_power)
        valley_filling_power = max(0, avg_power - valley_power)
        
        # 汇总信息
        summary = {
            'total_buy_hours': len(buy_advice),
            'total_sell_hours': len(sell_advice),
            'avg_predicted_power': float(avg_power),
            'peak_power': float(peak_power),
            'valley_power': float(valley_power),
            'peak_hours': self.PEAK_HOURS,
            'valley_hours': self.VALLEY_HOURS,
            'price_spread': self.TIME_OF_USE_PRICES['peak'] - self.TIME_OF_USE_PRICES['valley'],
        }
        
        if actual_power is not None:
            if len(actual_power) > 24:
                actual_hourly = np.array([
                    np.mean(actual_power[i*4:(i+1)*4]) for i in range(24)
                ])
            else:
                actual_hourly = actual_power[:24]
            summary['prediction_accuracy'] = float(1 - np.mean(np.abs(predicted_hourly - actual_hourly) / (actual_hourly + 1e-8)))
        
        return TradeOptimizationResult(
            buy_advice=buy_advice,
            sell_advice=sell_advice,
            expected_revenue=expected_revenue,
            cost_saving=cost_saving,
            peak_shaving_power=peak_shaving_power,
            valley_filling_power=valley_filling_power,
            summary=summary,
        )
    
    def to_dict(self, result: TradeOptimizationResult) -> Dict[str, Any]:
        """将结果转换为字典 - 确保所有值都是 JSON 可序列化的"""
        def safe_float(val, default=0.0):
            """安全转换为 float，处理 NaN/Inf"""
            import math
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return default
                return round(f, 2)
            except (TypeError, ValueError):
                return default
        
        return {
            'buy_advice': [
                {
                    'hour': int(a.hour),
                    'power': safe_float(a.power),
                    'price': safe_float(a.price, 0.504),
                    'revenue': safe_float(a.revenue),
                    'reason': str(a.reason),
                }
                for a in result.buy_advice
            ],
            'sell_advice': [
                {
                    'hour': int(a.hour),
                    'power': safe_float(a.power),
                    'price': safe_float(a.price, 0.704),
                    'revenue': safe_float(a.revenue),
                    'reason': str(a.reason),
                }
                for a in result.sell_advice
            ],
            'expected_revenue': safe_float(result.expected_revenue),
            'cost_saving': safe_float(result.cost_saving),
            'peak_shaving_power': safe_float(result.peak_shaving_power),
            'valley_filling_power': safe_float(result.valley_filling_power),
            'summary': {
                'total_buy_hours': int(result.summary.get('total_buy_hours', 0)),
                'total_sell_hours': int(result.summary.get('total_sell_hours', 0)),
                'avg_predicted_power': safe_float(result.summary.get('avg_predicted_power', 0)),
                'peak_power': safe_float(result.summary.get('peak_power', 0)),
                'valley_power': safe_float(result.summary.get('valley_power', 0)),
                'peak_hours': list(result.summary.get('peak_hours', [])),
                'valley_hours': list(result.summary.get('valley_hours', [])),
                'price_spread': safe_float(result.summary.get('price_spread', 0.2)),
                'peak_price': 0.704,
                'flat_price': 0.604,
                'valley_price': 0.50
