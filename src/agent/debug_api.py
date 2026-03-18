#!/usr/bin/env python3
"""
EcoLife Agent Debug API
提供智能助手的 API 接口供调试使用

使用方法:
    python -m src.agent.debug_api
    
    或访问:
    POST http://localhost:8000/chat - 发送聊天消息
    GET  http://localhost:8000/tools - 查看可用工具
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Global agent instance
_agent_executor = None
_thread_id = "debug_session"


def get_agent():
    """Get or create agent instance"""
    global _agent_executor
    
    if _agent_executor is not None:
        return _agent_executor
    
    try:
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_openai import ChatOpenAI
        
        from .tools import get_all_tools
        tools = get_all_tools()
        
        # Create LLM instance
        llm = ChatOpenAI(
            model="qwen3.5-plus",
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.7,
        )
        
        # Create memory saver
        memory = MemorySaver()
        
        # Create system message
        system_message = "你是一个专业的碳足迹助手，帮助用户查询碳排放、获取减碳建议。使用中文回复。"
        
        # Create agent
        _agent_executor = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
            checkpointer=memory,
        )
        
        return _agent_executor
        
    except Exception as e:
        print(f"Failed to create agent: {e}")
        return None


def chat(message: str, thread_id: str = None) -> dict:
    """
    Send a message to the agent and get response
    
    Args:
        message: User message
        thread_id: Optional thread ID for conversation memory
    
    Returns:
        dict: {"success": bool, "response": str, "error": str}
    """
    if thread_id:
        global _thread_id
        _thread_id = thread_id
    
    agent = get_agent()
    
    if agent is None:
        return {
            "success": False,
            "response": "Agent initialization failed",
            "error": "Failed to create agent instance"
        }
    
    try:
        from langchain_core.messages import HumanMessage
        
        config = {"configurable": {"thread_id": _thread_id}}
        response = agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        # Extract the last AI message
        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            output = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            output = "No response generated"
        
        return {
            "success": True,
            "response": output,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "response": None,
            "error": str(e)
        }


def list_tools() -> dict:
    """List all available tools"""
    try:
        from .tools import get_all_tools
        tools = get_all_tools()
        
        tool_list = []
        for tool in tools:
            tool_list.append({
                "name": tool.name,
                "description": tool.description
            })
        
        return {
            "success": True,
            "tools": tool_list
        }
        
    except Exception as e:
        return {
            "success": False,
            "tools": [],
            "error": str(e)
        }


class AgentAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Agent API"""
    
    def _send_response(self, data: dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._send_response({})
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/tools':
            result = list_tools()
            self._send_response(result)
        elif parsed_path.path == '/health':
            self._send_response({"status": "ok", "agent_ready": get_agent() is not None})
        else:
            self._send_response({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                thread_id = data.get('thread_id', 'default')
                
                if not message:
                    self._send_response({
                        "success": False,
                        "response": None,
                        "error": "Message is required"
                    }, 400)
                    return
                
                result = chat(message, thread_id)
                self._send_response(result)
                
            except json.JSONDecodeError:
                self._send_response({
                    "success": False,
                    "response": None,
                    "error": "Invalid JSON"
                }, 400)
        else:
            self._send_response({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        """Log HTTP requests"""
        print(f"[API] {args[0]}")


def run_server(host: str = 'localhost', port: int = 8000):
    """Run the API server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, AgentAPIHandler)
    print(f"Starting EcoLife Agent API server at http://{host}:{port}")
    print("Available endpoints:")
    print("  GET  /health     - Check server status")
    print("  GET  /tools      - List available tools")
    print("  POST /chat       - Send a message to the agent")
    print("\nExample usage:")
    print('  curl -X POST http://localhost:8000/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"message": "你好", "thread_id": "user1"}\'')
    print("\nPress Ctrl+C to stop the server")
    httpd.serve_forever()


if __name__ == '__main__':
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    run_server(host, port)