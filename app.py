from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.utils.logger import logger
from src.models.model_service import ModelService
from src.utils.env import detect_runtime_config
from src.utils.eta import ETAEstimator


st.set_page_config(
    page_title="Power Forecasting Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f5ef;
        --card: #ffffff;
        --ink: #172121;
        --accent: #d1495b;
        --accent-soft: #edae49;
        --line: #d0d3d4;
    }
    .stApp { background: radial-gradient(circle at 20% 10%, #fff9e6 0%, var(--bg) 45%, #eef4f3 100%); }
    .block-container { padding-top: 1.8rem; }
    .title-box {
        background: linear-gradient(110deg, var(--card), #fef3db);
        border: 1px solid var(--line);
        border-left: 8px solid var(--accent);
        border-radius: 14px;
        padding: 1.0rem 1.2rem;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #475467;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


AVAILABLE_MODELS = ["lstm", "gru", "xgboost", "moirai"]


@st.cache_resource
def get_runtime() -> Dict[str, Any]:
    runtime = detect_runtime_config()
    return {
        "is_linux": runtime.is_linux,
        "is_streamlit_cloud": runtime.is_streamlit_cloud,
        "lightweight_mode": runtime.lightweight_mode,
    }


@st.cache_resource
def get_model_service(data_path: str, model_dir: str, lookback: int) -> ModelService:
    return ModelService(data_path=data_path, model_dir=model_dir, lookback=lookback)


def _init_state() -> None:
    if "last_train_result" not in st.session_state:
        st.session_state["last_train_result"] = None
    if "last_predict_result" not in st.session_state:
        st.session_state["last_predict_result"] = None


def _config_panel() -> Dict[str, Any]:
    runtime = get_runtime()

    with st.sidebar:
        st.header("Runtime")
        st.caption("Cloud-aware deployment and model controls")

        data_path = st.text_input("Data CSV Path", value="data/data.csv")
        model_dir = st.text_input("Model Directory", value="models")
        lookback = st.slider("Lookback Window", min_value=12, max_value=192, value=24, step=12)

        st.divider()
        st.write("Environment")
        st.json(runtime)

        lightweight_ui = st.toggle(
            "Lightweight Mode",
            value=runtime["lightweight_mode"],
            help="Recommended on Streamlit Cloud to prevent OOM when using large foundation models.",
        )
        if lightweight_ui != runtime["lightweight_mode"]:
            st.info("Set LIGHTWEIGHT_MODE env variable for persistent runtime behavior.")

        st.divider()
        st.write("Model Selection")
        selected_models = st.multiselect(
            "Models",
            options=AVAILABLE_MODELS,
            default=["lstm", "xgboost", "moirai"],
            help="Select base models for training and comparison.",
        )

    return {
        "data_path": data_path,
        "model_dir": model_dir,
        "lookback": lookback,
        "selected_models": selected_models,
    }


def _show_header() -> None:
    st.markdown(
        """
        <div class="title-box">
            <h2 style="margin:0;">Multi-Model Forecasting Workbench</h2>
            <p class="small-note" style="margin:0.3rem 0 0 0;">
                LSTM / GRU / XGBoost / Moirai Zero-shot + Stacking Meta Learner
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _train_page(service: ModelService, selected_models: List[str]) -> None:
    st.subheader("Training Monitor")

    col_a, col_b = st.columns(2)
    with col_a:
        epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30)
    with col_b:
        batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=64)

    if st.button("Start Training", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model")
            return

        epoch_estimator = ETAEstimator(total_units=max(1, int(epochs)))
        model_estimator = ETAEstimator(total_units=max(1, len(selected_models)))
        model_estimator.start()

        progress_placeholder = st.empty()
        metric_placeholder = st.empty()

        state = {
            "current_model": "",
            "last_epoch": 0,
            "model_index": 0,
        }

        def on_progress(payload: Dict[str, Any]) -> None:
            model_name = payload.get("model", "unknown")
            epoch = int(payload.get("epoch", 0))
            total_epochs = int(payload.get("epochs", 1))
            model_idx = int(payload.get("model_index", 1))
            model_total = int(payload.get("model_total", 1))

            if model_name != state["current_model"]:
                state["current_model"] = model_name
                state["last_epoch"] = 0
                epoch_estimator.first_unit_duration = None
                epoch_estimator.start()

            if epoch == 1 and state["last_epoch"] == 0:
                epoch_estimator.observe_first_unit()
            state["last_epoch"] = epoch
            state["model_index"] = model_idx

            model_progress = (model_idx - 1) / max(model_total, 1)
            epoch_progress = epoch / max(total_epochs, 1)
            total_progress = min(0.999, model_progress + epoch_progress / max(model_total, 1))

            epoch_eta = epoch_estimator.format_seconds(epoch_estimator.estimate_remaining_seconds(epoch))
            models_eta = model_estimator.format_seconds(
                model_estimator.estimate_remaining_seconds(max(model_idx - 1, 0))
            )

            progress_placeholder.progress(
                total_progress,
                text=(
                    f"Model {model_idx}/{model_total}: {model_name.upper()} | "
                    f"Epoch {epoch}/{total_epochs} | ETA(epoch) {epoch_eta} | ETA(models) {models_eta}"
                ),
            )
            metric_placeholder.info(
                f"train_loss={payload.get('train_loss')} | val_loss={payload.get('val_loss')}"
            )

        with st.status("Training models...", expanded=True) as status_box:
            status_box.write("Preparing datasets and model registry...")
            try:
                result = service.train(
                    selected_models=selected_models,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    progress_callback=on_progress,
                )
                model_estimator.observe_first_unit()
                progress_placeholder.progress(1.0, text="Training completed")
                status_box.update(label="Training complete", state="complete")
                st.session_state["last_train_result"] = result
                st.success("Training finished and models saved")
            except Exception as exc:
                logger.exception("Training failed")
                status_box.update(label="Training failed", state="error")
                st.error(f"Training failed: {exc}")
                return

    if st.session_state.get("last_train_result") is not None:
        st.markdown("---")
        st.write("Latest training summary")
        st.json(st.session_state["last_train_result"])


def _prediction_page(service: ModelService, selected_models: List[str]) -> None:
    st.subheader("Prediction Comparison")

    horizon = st.slider("Prediction Horizon", min_value=24, max_value=288, value=96, step=24)
    compare_models = st.multiselect(
        "Compare Curves",
        options=selected_models + (["stacking"] if len(selected_models) >= 2 else []),
        default=selected_models[: min(3, len(selected_models))],
    )

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model")
            return
        try:
            result = service.predict(
                selected_models=selected_models,
                use_stacking=True,
                horizon=int(horizon),
            )
            st.session_state["last_predict_result"] = result
            st.success("Prediction completed")
        except Exception as exc:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")
            return

    pred_result = st.session_state.get("last_predict_result")
    if not pred_result:
        st.info("Run prediction to visualize model curves")
        return

    pred_map = pred_result.get("predictions", {})
    gt = pred_result.get("ground_truth", [])

    chart_df = pd.DataFrame()
    if gt:
        chart_df["ground_truth"] = pd.Series(gt)

    for name in compare_models:
        if name in pred_map:
            chart_df[name] = pd.Series(pred_map[name])

    if chart_df.empty:
        st.warning("No selected model outputs are available")
        return

    st.line_chart(chart_df, height=420)
    st.caption("You can overlay LSTM, XGBoost, and Moirai in one chart for direct visual comparison.")

    st.dataframe(chart_df.tail(30), use_container_width=True)


def _data_page(data_path: str) -> None:
    st.subheader("Data Snapshot")
    csv_path = Path(data_path)
    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing", int(df.isna().sum().sum()))

    st.dataframe(df.head(20), use_container_width=True)



def main() -> None:
    _init_state()
    _show_header()

    cfg = _config_panel()
    service = get_model_service(cfg["data_path"], cfg["model_dir"], cfg["lookback"])

    tabs = st.tabs(["Data", "Train", "Predict"])
    with tabs[0]:
        _data_page(cfg["data_path"])
    with tabs[1]:
        _train_page(service, cfg["selected_models"])
    with tabs[2]:
        _prediction_page(service, cfg["selected_models"])


if __name__ == "__main__":
    main()
