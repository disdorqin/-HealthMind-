"""SQLAlchemy ORM models.

This module defines core database tables for the system, such as
power load records and trading orders. These are minimal examples
that can be extended as the project evolves.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class PowerLoadRecord(Base):
    """Historical power load or generation record.

    Attributes:
        id: Primary key.
        timestamp: Datetime of the record.
        load: Power load value (MW).
        generation: Power generation value (MW).
        source_type: Type of generation source (e.g., wind, solar).
    """

    __tablename__ = "power_load_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    load: Mapped[float] = mapped_column(Float, nullable=True)
    generation: Mapped[float] = mapped_column(Float, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=True)


class TradeOrder(Base):
    """Electricity trade order.

    Attributes:
        id: Primary key.
        timestamp: Time the trade is created.
        direction: 'buy' or 'sell'.
        volume: Trade volume (MWh).
        price: Trade price (currency per MWh).
        pnl: Realized or expected profit and loss.
    """

    __tablename__ = "trade_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=True)

