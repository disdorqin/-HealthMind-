from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Legacy Frontend", page_icon="⚠️", layout="centered")

st.warning("src/frontend/app.py is now a legacy placeholder.")
st.info("Use the root app entry for production: streamlit run app.py")

st.markdown("""
### Migration Note
- Current production UI: root app.py
- Current model service: src/models/model_service.py
- Current data service: src/data/data_service.py

This file intentionally no longer depends on Flask APIs.
""")
