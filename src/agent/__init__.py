"""
EcoLife Agent Module
智能体模块，包含工具定义和 Agent 封装
"""

from .tools import (
    get_all_tools,
    get_current_carbon,
    predict_carbon,
    explain_carbon,
    get_recommendations,
    query_strategy,
)

from .agent import (
    create_agent_executor,
    run_agent,
    run_agent_stream,
)

__all__ = [
    # Tools
    "get_all_tools",
    "get_current_carbon",
    "predict_carbon",
    "explain_carbon",
    "get_recommendations",
    "query_strategy",
    # Agent
    "create_agent_executor",
    "run_agent",
    "run_agent_stream",
]
