"""后端 API - Flask RESTful 服务

可以通过以下方式启动：
1. python src/backend/api.py              # 直接运行
2. python main.py --mode api-only        # 通过main.py启动
3. python main.py --mode serve           # 作为完整系统的一部分启动
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app_layer import create_api_app
from src.core.utils.logger import logger


def main():
    """启动Flask应用"""
    app = create_api_app()
    
    logger.info("启动Flask API服务器...")
    logger.info("访问地址：http://localhost:5000")
    logger.info("API文档：http://localhost:5000/api/health")
    
    # 在生产环境中应使用WSGI服务器（如gunicorn）
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )


if __name__ == '__main__':
    main()


# 创建应用实例
app = create_app()


if __name__ == '__main__':
    logger.info("启动 Flask API 服务...")
    app.run(host='0.0.0.0', port=5000, debug=True)