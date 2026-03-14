"""Database schema definitions for core business models.

This module defines ORM models using SQLAlchemy 2.0 syntax for:
    - PowerWeatherModel
    - PredictionModel
    - PriceModel
    - TradeAnalysisModel

When executed as a script, it will create the corresponding tables in
the configured MySQL database using settings from the project .env
file and the SQLAlchemy engine defined in ``db_config.py``.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict

from sqlalchemy import DateTime, Float, Integer, String, Time
from sqlalchemy.orm import Mapped, mapped_column

from .db_config import engine  # ensure .env is loaded via db_config
from .models import Base


class PowerWeatherModel(Base):
    """Power and weather observation record.

    Attributes:
        id: Primary key.
        timestamp: Observation timestamp, indexed for fast lookup.
        actual_power: Measured power value (MW).
        wind_speed: Wind speed (m/s).
        temperature: Temperature (°C).
        irradiance: Solar irradiance (W/m²).
        is_holiday: Whether the day is a holiday.
        hour_of_day: Hour of day in 0–23.
    """

    __tablename__ = "power_weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    actual_power: Mapped[float] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float] = mapped_column(Float, nullable=True)
    temperature: Mapped[float] = mapped_column(Float, nullable=True)
    irradiance: Mapped[float] = mapped_column(Float, nullable=True)
    is_holiday: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a serializable dictionary.

        Returns:
            dict: Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actual_power": self.actual_power,
            "wind_speed": self.wind_speed,
            "temperature": self.temperature,
            "irradiance": self.irradiance,
            "is_holiday": bool(self.is_holiday),
            "hour_of_day": self.hour_of_day,
        }


class PredictionModel(Base):
    """Model prediction record.

    Attributes:
        id: Primary key.
        run_time: Time the prediction job was executed.
        target_timestamp: Target timestamp that the prediction refers to.
        predicted_power: Predicted power value (MW).
        model_type: Model type identifier (e.g., LSTM, XGBoost, Stacking).
        time_scale: Time scale of prediction (e.g., Day, Week, Month).
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    target_timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    predicted_power: Mapped[float] = mapped_column(Float, nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    time_scale: Mapped[str] = mapped_column(String(16), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a serializable dictionary.

        Returns:
            dict: Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "run_time": self.run_time.isoformat() if self.run_time else None,
            "target_timestamp": self.target_timestamp.isoformat()
            if self.target_timestamp
            else None,
            "predicted_power": self.predicted_power,
            "model_type": self.model_type,
            "time_scale": self.time_scale,
        }


class PriceModel(Base):
    """Time-of-use electricity price record.

    Attributes:
        id: Primary key.
        period_type: Period type (e.g., peak, flat, valley).
        start_time: Start time of the period during the day.
        end_time: End time of the period during the day.
        price: Price per MWh.
    """

    __tablename__ = "time_of_use_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    period_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a serializable dictionary.

        Returns:
            dict: Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "period_type": self.period_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "price": self.price,
        }


class TradeAnalysisModel(Base):
    """Trade optimization analysis record.

    Attributes:
        id: Primary key.
        run_time: Time the optimization analysis was executed.
        baseline_revenue: Revenue before optimization.
        optimized_revenue: Revenue after optimization.
        revenue_gain: Incremental revenue due to optimization.
        peak_shaving_energy: Energy used for peak shaving (MWh).
        valley_filling_energy: Energy used for valley filling (MWh).
        strategy_name: Optional strategy identifier.
    """

    __tablename__ = "trade_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    baseline_revenue: Mapped[float] = mapped_column(Float, nullable=False)
    optimized_revenue: Mapped[float] = mapped_column(Float, nullable=False)
    revenue_gain: Mapped[float] = mapped_column(Float, nullable=False)
    peak_shaving_energy: Mapped[float] = mapped_column(Float, nullable=True)
    valley_filling_energy: Mapped[float] = mapped_column(Float, nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a serializable dictionary.

        Returns:
            dict: Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "run_time": self.run_time.isoformat() if self.run_time else None,
            "baseline_revenue": self.baseline_revenue,
            "optimized_revenue": self.optimized_revenue,
            "revenue_gain": self.revenue_gain,
            "peak_shaving_energy": self.peak_shaving_energy,
            "valley_filling_energy": self.valley_filling_energy,
            "strategy_name": self.strategy_name,
        }


class TrainingMetricsModel(Base):
    """Model training metrics record.

    Attributes:
        id: Primary key.
        run_time: Time the model training was executed.
        model_name: Name of the trained model (e.g., LSTM, XGBoost, Stacking).
        model_type: Type of model (e.g., DL, Tree, Ensemble).
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        mape: Mean Absolute Percentage Error.
        r2: R² score.
        mse: Mean Squared Error.
        epochs: Number of training epochs (if applicable).
        batch_size: Batch size used during training.
        learning_rate: Learning rate used during training.
        training_time: Training time in seconds.
        dataset_size: Size of training dataset.
        validation_maemetrics: Validation metrics (JSON).
        test_metrics: Test metrics (JSON).
        version: Model version identifier.
        notes: Additional notes about the training.
    """

    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=True)
    r2: Mapped[float] = mapped_column(Float, nullable=True)
    mse: Mapped[float] = mapped_column(Float, nullable=True)
    epochs: Mapped[int] = mapped_column(Integer, nullable=True)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=True)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=True)
    training_time: Mapped[float] = mapped_column(Float, nullable=True)
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=True)
    validation_metrics: Mapped[str] = mapped_column(String(512), nullable=True)
    test_metrics: Mapped[str] = mapped_column(String(512), nullable=True)
    version: Mapped[str] = mapped_column(String(32), nullable=True)
    notes: Mapped[str] = mapped_column(String(256), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a serializable dictionary.

        Returns:
            dict: Dictionary representation of the record.
        """
        return {
            "id": self.id,
            "run_time": self.run_time.isoformat() if self.run_time else None,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2,
            "mse": self.mse,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "training_time": self.training_time,
            "dataset_size": self.dataset_size,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "version": self.version,
            "notes": self.notes,
        }


def create_all_tables() -> None:
    """Create all tables associated with the shared Base metadata.

    This function uses the SQLAlchemy engine configured in ``db_config``.
    """
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_all_tables()

