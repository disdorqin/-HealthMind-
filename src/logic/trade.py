"""交易优化逻辑 - 核心业务逻辑（无Flask依赖）"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np

from src.core.utils.logger import logger


class TimeOfUsePrice:
    """
    分时电价模型
    
    定义：
    - 高峰（Peak）：11-14点，18-23点 - 价格最高
    - 平段（Flat）：其他时段 - 价格中等
    - 低谷（Valley）：23-7点 - 价格最低
    """
    
    def __init__(self, peak_price: float = 1.2, flat_price: float = 0.8, valley_price: float = 0.4):
        """
        初始化分时电价
        
        Args:
            peak_price: 高峰电价（¥/kWh）
            flat_price: 平段电价（¥/kWh）
            valley_price: 低谷电价（¥/kWh）
        """
        self.peak_price = peak_price
        self.flat_price = flat_price
        self.valley_price = valley_price
    
    def get_price(self, hour: int) -> float:
        """获取特定小时的电价"""
        if 11 <= hour <= 14 or 18 <= hour <= 23:
            return self.peak_price
        elif 23 <= hour or hour < 7:
            return self.valley_price
        else:
            return self.flat_price
    
    def get_prices_24h(self) -> List[float]:
        """获取24小时电价"""
        return [self.get_price(h) for h in range(24)]
    
    def get_price_category(self, hour: int) -> str:
        """获取电价分类"""
        if 11 <= hour <= 14 or 18 <= hour <= 23:
            return 'PEAK'
        elif 23 <= hour or hour < 7:
            return 'VALLEY'
        else:
            return 'FLAT'


class TradeOptimizer:
    """交易优化器 - 完整的业务逻辑封装"""
    
    def __init__(self, predictions: np.ndarray, price_data: TimeOfUsePrice = None,
                 battery_capacity: float = 1000.0, battery_efficiency: float = 0.9):
        """
        初始化交易优化器
        
        Args:
            predictions: 24小时预测的功率值 (shape: (24,))
            price_data: 分时电价模型
            battery_capacity: 电池最大容量 (kWh)
            battery_efficiency: 电池效率（充放电往返效率）
        """
        if len(predictions) != 24:
            raise ValueError("预测数据应为24小时")
        
        self.predictions = np.array(predictions).reshape(24)
        self.price_data = price_data or TimeOfUsePrice()
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        
        logger.info(f"交易优化器初始化 - 预测值范围：{self.predictions.min():.2f}-{self.predictions.max():.2f}kW")
    
    def generate_trade_advice(self, threshold: float = None) -> Dict[str, Any]:
        """
        生成交易建议 - 基于电价和预测功率
        
        返回交易建议列表和经济指标
        """
        logger.info("生成交易建议")
        
        buy_advice = []
        sell_advice = []
        
        # 计算24小时的电价
        prices_24h = self.price_data.get_prices_24h()
        mean_price = np.mean(prices_24h)
        
        for hour in range(24):
            pred_power = float(self.predictions[hour])
            hour_price = prices_24h[hour]
            price_category = self.price_data.get_price_category(hour)
            
            # 买入建议：低谷时段 + 功率相对较低
            if price_category == 'VALLEY' and hour_price <= mean_price:
                priority = int(10 - (hour_price / mean_price) * 5)
                buy_advice.append({
                    'hour': hour,
                    'hour_str': f'{hour:02d}:00',
                    'power': pred_power,
                    'price': float(hour_price),
                    'reason': '低谷时段，电价最低，适合充电储能',
                    'priority': max(1, min(10, priority))
                })
            
            # 卖出建议：高峰时段 + 功率相对较高
            if price_category == 'PEAK' and hour_price >= mean_price:
                priority = int((hour_price / mean_price) * 10)
                sell_advice.append({
                    'hour': hour,
                    'hour_str': f'{hour:02d}:00',
                    'power': pred_power,
                    'price': float(hour_price),
                    'reason': '高峰时段，电价最高，适合放电自用',
                    'priority': max(1, min(10, priority))
                })
        
        # 按优先级排序
        buy_advice.sort(key=lambda x: x['priority'], reverse=True)
        sell_advice.sort(key=lambda x: x['priority'], reverse=True)
        
        # 计算汇总信息
        total_buy_power = sum(a['power'] for a in buy_advice)
        total_sell_power = sum(a['power'] for a in sell_advice)
        buy_price_avg = np.mean([a['price'] for a in buy_advice]) if buy_advice else 0
        sell_price_avg = np.mean([a['price'] for a in sell_advice]) if sell_advice else 0
        
        # 计算经济效益
        purchase_cost = total_buy_power * buy_price_avg * 0.95
        sell_revenue = total_sell_power * sell_price_avg * 0.85
        expected_revenue = max(0, sell_revenue - purchase_cost)
        cost_saving = total_buy_power * (mean_price - buy_price_avg)
        
        logger.info(f"✓ 交易建议生成完成 - 买入{len(buy_advice)}条，卖出{len(sell_advice)}条")
        
        return {
            'status': 'success',
            'buy_advice': buy_advice[:10],
            'sell_advice': sell_advice[:10],
            'summary': {
                'peak_price': float(self.price_data.peak_price),
                'flat_price': float(self.price_data.flat_price),
                'valley_price': float(self.price_data.valley_price),
                'mean_price': float(mean_price),
                'total_buy_power': float(total_buy_power),
                'total_sell_power': float(total_sell_power),
                'buy_price_avg': float(buy_price_avg),
                'sell_price_avg': float(sell_price_avg),
            },
            'expected_revenue': float(expected_revenue),
            'cost_saving': float(cost_saving),
            'peak_shaving_power': float(np.percentile(self.predictions, 90)),
            'valley_filling_power': float(np.percentile(self.predictions, 10)),
        }
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """计算交易指标"""
        prices_24h = self.price_data.get_prices_24h()
        
        daily_energy = float(np.sum(self.predictions))
        weighted_cost = float(np.sum(self.predictions * prices_24h) / daily_energy) if daily_energy > 0 else 0
        revenue_potential = float(np.sum(self.predictions) * self.price_data.peak_price * 0.8)
        
        peak_hours = [h for h in range(24) if self.price_data.get_price_category(h) == 'PEAK']
        peak_power = np.sum(self.predictions[peak_hours]) if peak_hours else 0
        peak_shaving_benefit = float(peak_power * (self.price_data.peak_price - self.price_data.flat_price))
        
        valley_hours = [h for h in range(24) if self.price_data.get_price_category(h) == 'VALLEY']
        valley_power = np.sum(self.predictions[valley_hours]) if valley_hours else 0
        valley_filling_benefit = float(valley_power * (self.price_data.flat_price - self.price_data.valley_price))
        
        return {
            'status': 'success',
            'daily_energy': daily_energy,
            'average_cost_per_kwh': weighted_cost,
            'revenue_potential': revenue_potential,
            'peak_shaving_benefit': peak_shaving_benefit,
            'valley_filling_benefit': valley_filling_benefit,
            'total_benefit': peak_shaving_benefit + valley_filling_benefit
        }
    
    def analyze_risk(self) -> Dict[str, Any]:
        """分析交易风险"""
        logger.info("分析交易风险")
        
        mean_val = float(np.mean(self.predictions))
        std_val = float(np.std(self.predictions))
        cv = float(std_val / mean_val) if mean_val != 0 else 0
        
        min_val = float(np.min(self.predictions))
        max_val = float(np.max(self.predictions))
        range_val = max_val - min_val
        
        q25 = float(np.percentile(self.predictions, 25))
        q50 = float(np.percentile(self.predictions, 50))
        q75 = float(np.percentile(self.predictions, 75))
        iqr = q75 - q25
        
        volatility_score = min(10, cv * 20)
        range_score = min(10, range_val / mean_val * 5) if mean_val > 0 else 0
        risk_score = (volatility_score * 0.6 + range_score * 0.4)
        
        if risk_score < 2:
            risk_level = 'VERY_LOW'
            recommendation = '风险极低，可放心操作'
        elif risk_score < 4:
            risk_level = 'LOW'
            recommendation = '风险较低，建议积极操作'
        elif risk_score < 6:
            risk_level = 'MEDIUM'
            recommendation = '风险中等，建议谨慎操作'
        elif risk_score < 8:
            risk_level = 'HIGH'
            recommendation = '风险较高，建议保守操作'
        else:
            risk_level = 'VERY_HIGH'
            recommendation = '风险极高，建议暂停操作'
        
        logger.info(f"✓ 风险分析完成 - 等级：{risk_level}，分数：{risk_score:.2f}")
        
        return {
            'status': 'success',
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'statistics': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'q25': q25,
                'q50': q50,
                'q75': q75,
                'iqr': iqr,
            },
            'risk_indicators': {
                'volatility_score': float(volatility_score),
                'range_score': float(range_score),
                'coefficient_of_variation': cv,
            }
        }
