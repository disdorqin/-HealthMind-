#!/usr/bin/env python3
"""
阿里云通义千问大模型客户端封装
提供统一的 LLM 调用接口
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def get_aliyun_llm(model: str = "qwen-plus", **kwargs):
    """
    获取配置好的阿里云通义千问 LLM 实例
    
    Args:
        model: 模型名称，可选：
            - qwen-turbo: 通义千问极速版
            - qwen-plus: 通义千问 Plus（推荐）
            - qwen-max: 通义千问 Max（最强）
        **kwargs: 其他参数传递给 ChatTongyi
        
    Returns:
        ChatTongyi 实例，如果初始化失败则返回 None
        
    Example:
        >>> llm = get_aliyun_llm()
        >>> response = llm.invoke("你好")
        >>> print(response.content)
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not found in environment variables")
        print("Please add to .env: DASHSCOPE_API_KEY=your_api_key")
        return None
    
    try:
        from langchain_community.chat_models import ChatTongyi
        
        llm = ChatTongyi(
            model=model,
            dashscope_api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        
        return llm
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install: pip install langchain-community dashscope")
        return None
    except Exception as e:
        print(f"Failed to initialize ChatTongyi: {e}")
        return None


def call_aliyun_llm(prompt: str, model: str = "qwen-plus", **kwargs) -> Optional[str]:
    """
    直接调用阿里云通义千问 LLM 并返回回复内容
    
    Args:
        prompt: 用户输入的问题或提示
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        模型回复的文本内容，如果调用失败则返回 None
        
    Example:
        >>> response = call_aliyun_llm("你好，请介绍一下你自己")
        >>> print(response)
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not found")
        return None
    
    try:
        import dashscope
        
        response = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            result_format="message",
            api_key=api_key,
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"API Error: {response.code} - {response.message}")
            return None
            
    except Exception as e:
        print(f"Call failed: {e}")
        return None


# 测试函数
if __name__ == "__main__":
    print("="*60)
    print("Testing Aliyun Qwen LLM Client")
    print("="*60)
    
    # 测试方法 1：使用 ChatTongyi
    print("\nMethod 1: ChatTongyi (LangChain)")
    print("-"*60)
    llm = get_aliyun_llm()
    if llm:
        try:
            response = llm.invoke("你好，请介绍一下你自己")
            print(f"Response: {response.content[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Failed to initialize ChatTongyi")
    
    # 测试方法 2：直接调用
    print("\nMethod 2: Direct API call")
    print("-"*60)
    response = call_aliyun_llm("你好，请介绍一下你自己")
    if response:
        print(f"Response: {response[:200]}...")
    else:
        print("API call failed")
    
    print("\n" + "="*60)