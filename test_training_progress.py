"""Smoke tests for the new multi-model training and prediction pipeline."""

from __future__ import annotations

from pathlib import Path

from src.models.model_service import ModelService


def test_model_service_smoke() -> None:
    service = ModelService(
        data_path="data/data.csv",
        model_dir="models/test_artifacts",
        lookback=24,
    )

    train_result = service.train(
        selected_models=["xgboost", "moirai"],
        epochs=1,
        batch_size=64,
    )

    assert "models" in train_result
    assert "xgboost" in train_result["models"]
    assert "moirai" in train_result["models"]
    assert train_result["stacking"]["enabled"] is True

    predict_result = service.predict(
        selected_models=["xgboost", "moirai"],
        use_stacking=True,
        horizon=24,
    )

    preds = predict_result.get("predictions", {})
    assert "xgboost" in preds
    assert "moirai" in preds
    assert "stacking" in preds
    assert len(preds["stacking"]) > 0


def cleanup_test_artifacts() -> None:
    artifacts = Path("models/test_artifacts")
    if artifacts.exists():
        for file in artifacts.glob("*"):
            if file.is_file():
                file.unlink()


if __name__ == "__main__":
    test_model_service_smoke()
    cleanup_test_artifacts()
    print("ModelService smoke test passed")
