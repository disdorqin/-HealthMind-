from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    is_linux: bool
    is_streamlit_cloud: bool
    lightweight_mode: bool


def detect_runtime_config() -> RuntimeConfig:
    """Detect runtime and pick safe defaults for Streamlit Cloud."""
    is_linux = platform.system().lower() == "linux"
    cloud_markers = [
        os.getenv("STREAMLIT_RUNTIME"),
        os.getenv("STREAMLIT_SERVER_PORT"),
        os.getenv("STREAMLIT_SHARING_MODE"),
    ]
    is_streamlit_cloud = is_linux and any(marker is not None for marker in cloud_markers)

    # Manual override wins. Useful for debugging cloud-like behavior locally.
    env_lightweight = os.getenv("LIGHTWEIGHT_MODE", "").strip().lower()
    if env_lightweight in {"1", "true", "yes", "on"}:
        lightweight_mode = True
    elif env_lightweight in {"0", "false", "no", "off"}:
        lightweight_mode = False
    else:
        # Default to lightweight on cloud to avoid OOM for large foundation models.
        lightweight_mode = is_streamlit_cloud

    return RuntimeConfig(
        is_linux=is_linux,
        is_streamlit_cloud=is_streamlit_cloud,
        lightweight_mode=lightweight_mode,
    )
