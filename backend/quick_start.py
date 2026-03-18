#!/usr/bin/env python3
"""
快速启动脚本 - 用于测试后端 API 是否正常

使用方法:
    python backend/quick_start.py

启动后会显示:
    * Running on http://127.0.0.1:5000
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.api import create_app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "="*50)
    print("🚀 EcoLife 后端 API 启动中...")
    print("="*50)
    print("\n📡 API 地址：http://127.0.0.1:5000")
    print("🔍 健康检查：http://127.0.0.1:5000/api/health")
    print("\n💡 提示：")
    print("   - 按 Ctrl+C 停止服务")
    print("   - 启动后再运行前端：streamlit run app.py")
    print("="*50 + "\n")
    
    app.run(host="127.0.0.1", port=5000, debug=True)