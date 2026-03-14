"""Legacy backend package exports.

This package is kept for compatibility only. The default production path is
the Streamlit app entry at project root.
"""

from src.backend.api import main

__all__ = ["main"]