from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    runs: Mapped[list["PredictionRun"]] = relationship(back_populates="model_metadata")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    run_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    production_day: Mapped[date | None] = mapped_column(Date, nullable=True)

    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    predicted_defect_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_defect_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_error_pp: Mapped[float | None] = mapped_column(Float, nullable=True)
    flagged_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flagged_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_k_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_metadata_id: Mapped[int] = mapped_column(
        ForeignKey("model_metadata.id", ondelete="RESTRICT"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_metadata: Mapped[ModelMetadata] = relationship(back_populates="runs")
    results: Mapped[list["PredictionResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    row_id: Mapped[str] = mapped_column(String(200), nullable=False)
    production_day: Mapped[date | None] = mapped_column(Date, nullable=True)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prediction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_top_k: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[PredictionRun] = relationship(back_populates="results")
