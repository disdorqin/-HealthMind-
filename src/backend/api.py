"""Legacy Flask API entry.

This project now uses Streamlit + direct ModelService calls as the default
production path. Flask is kept only for legacy compatibility.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app_layer import create_api_app
from src.core.utils.logger import logger


def main():
    """Start Flask app only when explicitly enabled."""
    if os.getenv("ENABLE_LEGACY_FLASK", "0").strip() not in {"1", "true", "True"}:
        logger.error("Legacy Flask API is disabled by default.")
        logger.error("Use Streamlit app entry instead: streamlit run app.py")
        logger.error("Set ENABLE_LEGACY_FLASK=1 to force-start this legacy server.")
        return

    app = create_api_app()

    logger.warning("Starting legacy Flask API server")
    logger.warning("Default production entry is: streamlit run app.py")
    logger.info("Legacy API address: http://localhost:5000")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
