#!/usr/bin/env python3
"""
EcoLife Agent Wrapper
提供智能助手的核心功能 - 使用 LangGraph 新 API
"""

import os
from typing import Generator, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# OpenAI compatible endpoint
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def create_agent_executor():
    """
    Create and return an Agent executor instance using LangGraph
    
    Returns:
        AgentExecutor: LangGraph compiled agent graph
    """
    if not DASHSCOPE_API_KEY:
        print("Warning: DASHSCOPE_API_KEY not found")
        return None
    
    try:
        # Import LangGraph prebuilt agent
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        
        # Use ChatOpenAI with DashScope compatible endpoint
        from langchain_openai import ChatOpenAI
        
        # Import EcoLife tools
        from .tools import get_all_tools
        tools = get_all_tools()
        
        # Create LLM instance using OpenAI compatible API
        llm = ChatOpenAI(
            model="qwen3.5-plus",
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.7,
        )
        
        # Create memory saver for conversation history
        memory = MemorySaver()
        
        # Create system message
        system_message = "你是一个专业的碳足迹助手，帮助用户查询碳排放、获取减碳建议。使用中文回复。你有以下工具可用：get_current_carbon（获取当前碳排放）、predict_carbon（预测碳排放）、explain_carbon（解释碳排放构成）、get_recommendations（获取减碳建议）、query_strategy（查询减碳策略）。"
        
        # Create ReAct agent using LangGraph prebuilt
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
            checkpointer=memory,
        )
        
        return agent
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install: pip install langgraph langchain-openai")
        return None
    except Exception as e:
        print(f"Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_agent(user_input: str, agent_executor=None, thread_id: str = "default") -> str:
    """
    Run Agent and return response
    
    Args:
        user_input (str): User input
        agent_executor: Optional Agent executor instance
        thread_id: Conversation thread ID for memory
        
    Returns:
        str: Agent response
    """
    if agent_executor is None:
        agent_executor = create_agent_executor()
    
    if agent_executor is None:
        return "抱歉，智能助手服务暂时不可用，请稍后再试。"
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Use LangGraph invocation with config for thread
        config = {"configurable": {"thread_id": thread_id}}
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        # Extract the last AI message from messages list
        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return "抱歉，我没有理解你的问题。"
    except Exception as e:
        return f"抱歉，处理您的请求时出现错误：{str(e)}"


def run_agent_stream(user_input: str, agent_executor=None, thread_id: str = "default") -> Generator[str, None, None]:
    """
    Run Agent in streaming mode and return response incrementally
    
    Args:
        user_input (str): User input
        agent_executor: Optional Agent executor instance
        thread_id: Conversation thread ID for memory
        
    Yields:
        str: Text chunks
    """
    if agent_executor is None:
        agent_executor = create_agent_executor()
    
    if agent_executor is None:
        yield "抱歉，智能助手服务暂时不可用，请稍后再试。"
        return
    
    try:
        from langchain_core.messages import HumanMessage
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use streaming invocation
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        ):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, 'content'):
                        yield msg.content
                    else:
                        yield str(msg)
    except Exception as e:
        yield f"抱歉，处理您的请求时出现错误：{str(e)}"


# Test function
if __name__ == "__main__":
    print("="*60)
    print("EcoLife Agent Test")
    print("="*60)
    
    # Create Agent
    agent = create_agent_executor()
    
    if agent:
        print("\nAgent created successfully!")
        
        # Test conversation
        test_questions = [
            "你好！",
            "我今天碳排多少？",
            "有什么减碳建议吗？",
        ]
        
        for question in test_questions:
            print(f"\nUser: {question}")
            response = run_agent(question, agent)
            print(f"Assistant: {response}")
    else:
        print("\nFailed to create agent")
        print("Please check:")
        print("1. DASHSCOPE_API_KEY is set in .env")
        print("2. All dependencies are installed")
    
    print("\n" + "="*60)