#!/usr/bin/env python3
"""
EcoLife Agent Tools
定义智能体可调用的工具函数
"""

from datetime import datetime
from typing import Optional
from langchain.tools import tool

# 导入 EcoLife 服务
from src.services.carbon_engine import CarbonEngine
from src.services.prediction_service import PredictionService
from src.services.trade_service import HorizonTradeAdvisor


# 初始化服务实例
carbon_engine = CarbonEngine(baseline_kg=12.5)
prediction_service = PredictionService()
trade_service = HorizonTradeAdvisor()


@tool
def get_current_carbon() -> dict:
    """获取用户今日已产生的总碳排放量（kg CO₂e）。
    
    Returns:
        dict: {"carbon": float} - 碳排放量（kg CO₂e）
    """
    try:
        # 调用 carbon_engine 获取今日总排放
        carbon_value = carbon_engine.get_today_total()
        return {"carbon": float(carbon_value)}
    except Exception as e:
        return {"error": f"获取碳排放量失败：{str(e)}"}


@tool
def predict_carbon(date: str, behavior: Optional[str] = None) -> dict:
    """预测特定日期在给定行为下的碳排放量。
    
    Args:
        date (str): 日期，格式 YYYY-MM-DD
        behavior (str, optional): 描述行为的文本，如"开车上班 + 吃牛肉"
        
    Returns:
        dict: {"predicted": float, "unit": "kg CO₂e"} - 预测碳排放量
    """
    try:
        # 验证日期格式
        datetime.strptime(date, "%Y-%m-%d")
        
        # 调用预测服务
        if behavior:
            # 如果有行为描述，使用行为预测
            predicted_value = prediction_service.predict_with_behavior(date, behavior)
        else:
            # 否则使用默认预测
            predicted_value = prediction_service.predict(date)
        
        return {"predicted": float(predicted_value), "unit": "kg CO₂e"}
    except ValueError as e:
        return {"error": f"日期格式错误，请使用 YYYY-MM-DD 格式：{str(e)}"}
    except Exception as e:
        return {"error": f"预测失败：{str(e)}"}


@tool
def explain_carbon(date: str) -> dict:
    """解释某一天碳排放构成的主要因素，基于 SHAP 归因。
    
    Args:
        date (str): 日期，格式 YYYY-MM-DD
        
    Returns:
        dict: {"factors": dict, "total": float} - 各因素贡献及总量
    """
    try:
        # 验证日期格式
        datetime.strptime(date, "%Y-%m-%d")
        
        # 调用 carbon_engine 获取 SHAP 解释
        factors = carbon_engine.get_shap_explanation(date)
        total = sum(factors.values()) if isinstance(factors, dict) else 0
        
        return {"factors": factors, "total": float(total)}
    except ValueError as e:
        return {"error": f"日期格式错误，请使用 YYYY-MM-DD 格式：{str(e)}"}
    except Exception as e:
        return {"error": f"获取解释失败：{str(e)}"}


@tool
def get_recommendations(date: str) -> dict:
    """获取指定日期的个性化减碳建议清单。
    
    Args:
        date (str): 日期，格式 YYYY-MM-DD
        
    Returns:
        dict: {"recommendations": list} - 建议列表
    """
    try:
        # 验证日期格式
        datetime.strptime(date, "%Y-%m-%d")
        
        # 调用 trade_service 获取建议
        recommendations = trade_service.get_daily_advice(date)
        
        # 确保返回的是列表
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        elif not isinstance(recommendations, list):
            recommendations = list(recommendations)
        
        return {"recommendations": recommendations}
    except ValueError as e:
        return {"error": f"日期格式错误，请使用 YYYY-MM-DD 格式：{str(e)}"}
    except Exception as e:
        return {"error": f"获取建议失败：{str(e)}"}


@tool
def query_strategy(scene: str) -> dict:
    """查询特定生活场景的减碳策略库。
    
    Args:
        scene (str): 场景关键词，如"通勤"、"饮食"、"能耗"
        
    Returns:
        dict: {"strategies": list} - 策略列表，每项包含 action 和 reduction
    """
    try:
        # 调用 carbon_engine 查询策略
        strategies = carbon_engine.query_strategy(scene)
        
        # 确保返回格式统一
        if isinstance(strategies, dict):
            strategies = [strategies]
        elif not isinstance(strategies, list):
            strategies = [{"action": str(strategies), "reduction": 0}]
        
        return {"strategies": strategies}
    except Exception as e:
        return {"error": f"查询策略失败：{str(e)}"}


# 获取所有工具的函数
def get_all_tools():
    """获取所有可用的工具列表。
    
    Returns:
        list: 包含所有工具函数的列表
    """
    return [
        get_current_carbon,
        predict_carbon,
        explain_carbon,
        get_recommendations,
        query_strategy,
    ]


# 测试函数
if __name__ == "__main__":
    print("="*60)
    print("EcoLife Agent Tools Test")
    print("="*60)
    
    # 测试每个工具
    tools = get_all_tools()
    
    for tool_func in tools:
        print(f"\nTesting: {tool_func.name}")
        print(f"Description: {tool_func.description}")
        
        # 根据工具类型调用测试
        if tool_func.name == "get_current_carbon":
            result = tool_func.func()
            print(f"Result: {result}")
        elif tool_func.name == "get_recommendations":
            today = datetime.now().strftime("%Y-%m-%d")
            result = tool_func.func(today)
            print(f"Result: {result}")
        elif tool_func.name == "query_strategy":
            result = tool_func.func("通勤")
            print(f"Result: {result}")
        else:
            print("Skipped (requires specific parameters)")
    
    print("\n" + "="*60)