from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from MODEL_WORK import predict_day, predict_random_row

from app.db import SessionLocal, engine
from app.models import Base, ModelMetadata, PredictionResult, PredictionRun, utc_now
from app.runtime import available_days, get_bundle, get_dataset, validate_runtime_files


class ServiceError(Exception):
    pass


class NotFoundError(ServiceError):
    pass


class ValidationError(ServiceError):
    pass


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _parse_day(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _bundle_model_version(bundle: dict[str, Any]) -> str:
    """Получить единую версию активного model bundle."""

    return str(
        bundle.get(
            "model_version",
            bundle.get("bundle_version", "1.0"),
        )
    )


def init_database() -> None:
    """Проверить runtime-файлы, создать таблицы и синхронизировать metadata."""

    validate_runtime_files()
    Base.metadata.create_all(bind=engine)

    bundle = get_bundle()
    model_version = _bundle_model_version(bundle)

    with SessionLocal() as session:
        existing = session.execute(
            select(ModelMetadata).where(
                ModelMetadata.model_version == model_version
            )
        ).scalar_one_or_none()

        if existing is not None:
            # Версия является ключом связи запуска с реально загруженным bundle.
            # На случай повторного старта синхронизируем отображаемые поля.
            existing.model_name = str(
                bundle.get("model_name", "CatBoost defect model")
            )
            existing.threshold = float(bundle["threshold_raw"])
            existing.feature_count = int(bundle["feature_count"])
            existing.feature_names = list(bundle["feature_names"])
            session.commit()
            return

        metadata = ModelMetadata(
            model_name=str(
                bundle.get("model_name", "CatBoost defect model")
            ),
            model_version=model_version,
            threshold=float(bundle["threshold_raw"]),
            feature_count=int(bundle["feature_count"]),
            feature_names=list(bundle["feature_names"]),
        )
        session.add(metadata)

        try:
            session.commit()
        except IntegrityError:
            # Возможен одновременный старт API и нескольких worker-контейнеров.
            # Уникальный model_version не позволяет создать дубликат.
            session.rollback()


def get_model_metadata(session: Session) -> ModelMetadata:
    """Вернуть metadata именно для текущего загруженного bundle."""

    bundle = get_bundle()
    model_version = _bundle_model_version(bundle)

    item = session.execute(
        select(ModelMetadata).where(
            ModelMetadata.model_version == model_version
        )
    ).scalar_one_or_none()

    if item is None:
        raise RuntimeError(
            "Метаданные активной модели не инициализированы: "
            f"model_version={model_version}"
        )

    return item


def serialize_model_metadata(item: ModelMetadata) -> dict[str, Any]:
    bundle = get_bundle()
    calibration = bundle.get("calibration", {})
    calibrated_reference = bundle.get("threshold_calibrated_reference")

    return {
        "model_name": item.model_name,
        "model_version": item.model_version,
        # Порог применяется к raw_score.
        "threshold": item.threshold,
        "threshold_type": "raw_score",
        "decision_rule": "raw_score >= threshold",
        # Справочное значение не используется для бинарного решения сервиса.
        "threshold_calibrated_reference": (
            None
            if calibrated_reference is None
            else float(calibrated_reference)
        ),
        "feature_count": item.feature_count,
        "feature_names": item.feature_names,
        "calibration_enabled": bool(
            calibration.get(
                "enabled",
                bundle.get("calibrator") is not None,
            )
        ),
        "probability_type": "calibrated_probability",
        "daily_rate_rule": "mean(calibrated_probability) * 100",
        "top_k_share": float(bundle.get("top_k_share", 0.10)),
        "created_at": item.created_at,
    }


def serialize_run(item: PredictionRun) -> dict[str, Any]:
    return {
        "id": item.id,
        "task_id": item.task_id,
        "run_type": item.run_type,
        "status": item.status,
        "production_day": (
            item.production_day.isoformat()
            if item.production_day
            else None
        ),
        "row_count": item.row_count,
        "predicted_defect_rate": item.predicted_defect_rate,
        "actual_defect_rate": item.actual_defect_rate,
        "absolute_error_pp": item.absolute_error_pp,
        "flagged_count": item.flagged_count,
        "flagged_share": item.flagged_share,
        "top_k_count": item.top_k_count,
        "worker_id": item.worker_id,
        "error_message": item.error_message,
        "created_at": item.created_at,
        "started_at": item.started_at,
        "completed_at": item.completed_at,
    }


def serialize_result(item: PredictionResult) -> dict[str, Any]:
    return {
        "id": item.id,
        "run_id": item.run_id,
        "row_id": item.row_id,
        "production_day": (
            item.production_day.isoformat()
            if item.production_day
            else None
        ),
        "raw_score": item.raw_score,
        "probability": item.probability,
        "predicted_class": item.predicted_class,
        "actual_class": item.actual_class,
        "prediction_correct": item.prediction_correct,
        "risk": "Высокий" if item.predicted_class == 1 else "Обычный",
        "risk_rank": item.risk_rank,
        "is_top_k": item.is_top_k,
    }


def _prediction_result_from_row(
    run_id: int,
    row: pd.Series,
) -> PredictionResult:
    return PredictionResult(
        run_id=run_id,
        row_id=str(row.get("row_id", "unknown")),
        production_day=_parse_day(row.get("production_day")),
        raw_score=float(row["raw_score"]),
        probability=float(row["defect_probability"]),
        predicted_class=int(row["predicted_class"]),
        actual_class=_int_or_none(row.get("actual_target")),
        prediction_correct=_bool_or_none(row.get("prediction_correct")),
        risk_rank=_int_or_none(row.get("risk_rank")),
        is_top_k=bool(row.get("is_top_k", False)),
    )


def create_single_prediction(
    session: Session,
) -> tuple[PredictionRun, PredictionResult]:
    metadata = get_model_metadata(session)

    run = PredictionRun(
        task_id=str(uuid.uuid4()),
        run_type="single",
        status="processing",
        model_metadata_id=metadata.id,
        started_at=utc_now(),
    )
    session.add(run)
    session.flush()

    try:
        result_df = predict_random_row(get_dataset(), get_bundle())
        row = result_df.iloc[0]
        result = _prediction_result_from_row(run.id, row)

        # Для одиночного запуска одна строка одновременно является Top-K.
        result.risk_rank = 1
        result.is_top_k = True
        session.add(result)

        # Для n=1 агрегаты рассчитываются той же формулой, что и для дня.
        run.status = "completed"
        run.production_day = result.production_day
        run.row_count = 1
        run.predicted_defect_rate = float(result.probability * 100)
        run.actual_defect_rate = (
            None
            if result.actual_class is None
            else float(result.actual_class * 100)
        )
        run.absolute_error_pp = (
            None
            if run.actual_defect_rate is None
            else abs(
                run.predicted_defect_rate
                - run.actual_defect_rate
            )
        )
        run.flagged_count = int(result.predicted_class)
        run.flagged_share = float(result.predicted_class)
        run.top_k_count = 1
        run.completed_at = utc_now()

        session.commit()
        session.refresh(run)
        session.refresh(result)
        return run, result

    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = utc_now()
        session.commit()
        raise


def create_daily_run(
    session: Session,
    production_day: str | date,
) -> PredictionRun:
    try:
        parsed_day = pd.to_datetime(
            production_day,
            errors="raise",
        )

        if pd.isna(parsed_day):
            raise ValueError("Дата пустая")

        requested_day = pd.Timestamp(parsed_day).strftime("%Y-%m-%d")

    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Некорректная производственная дата. "
            "Ожидается формат YYYY-MM-DD."
        ) from exc

    day_map = {
        item["production_day"]: item["row_count"]
        for item in available_days()
    }

    if requested_day not in day_map:
        raise NotFoundError(
            f"За день {requested_day} строки не найдены"
        )

    metadata = get_model_metadata(session)

    run = PredictionRun(
        task_id=str(uuid.uuid4()),
        run_type="daily",
        status="queued",
        production_day=pd.Timestamp(requested_day).date(),
        row_count=int(day_map[requested_day]),
        model_metadata_id=metadata.id,
    )

    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def mark_run_failed(
    session: Session,
    run: PredictionRun,
    error: Exception | str,
) -> None:
    run.status = "failed"
    run.error_message = str(error)
    run.completed_at = utc_now()
    session.add(run)
    session.commit()


def process_daily_run(
    run_id: int,
    worker_id: str,
) -> PredictionRun:
    with SessionLocal() as session:
        run = session.get(PredictionRun, run_id)

        if run is None:
            raise NotFoundError(f"Запуск {run_id} не найден")

        if run.status == "completed":
            return run

        run.status = "processing"
        run.worker_id = worker_id
        run.started_at = utc_now()
        run.error_message = None
        session.commit()

        try:
            day_output = predict_day(
                get_dataset(),
                production_day=run.production_day.isoformat(),
                bundle=get_bundle(),
            )
            summary = day_output["summary"]
            results_df = day_output["results"]

            session.execute(
                delete(PredictionResult).where(
                    PredictionResult.run_id == run.id
                )
            )
            session.add_all(
                [
                    _prediction_result_from_row(run.id, row)
                    for _, row in results_df.iterrows()
                ]
            )

            run.status = "completed"
            run.row_count = int(summary["row_count"])
            run.predicted_defect_rate = _float_or_none(
                summary["predicted_daily_rate"]
            )
            run.actual_defect_rate = _float_or_none(
                summary.get("actual_daily_rate")
            )
            run.absolute_error_pp = _float_or_none(
                summary.get("absolute_error_pp")
            )
            run.flagged_count = int(summary["flagged_count"])
            run.flagged_share = float(summary["flagged_share"])
            run.top_k_count = int(summary["top_k_count"])
            run.completed_at = utc_now()

            session.commit()
            session.refresh(run)
            return run

        except Exception as exc:
            session.rollback()
            run = session.get(PredictionRun, run_id)

            if run is not None:
                mark_run_failed(session, run, exc)

            raise


def get_run(
    session: Session,
    run_id: int,
) -> PredictionRun:
    item = session.get(PredictionRun, run_id)

    if item is None:
        raise NotFoundError(f"Запуск {run_id} не найден")

    return item


def list_runs(
    session: Session,
    limit: int = 50,
) -> list[PredictionRun]:
    return list(
        session.execute(
            select(PredictionRun)
            .order_by(
                PredictionRun.created_at.desc(),
                PredictionRun.id.desc(),
            )
            .limit(limit)
        ).scalars().all()
    )


def list_results(
    session: Session,
    run_id: int,
    top_only: bool = False,
    limit: int = 5000,
) -> list[PredictionResult]:
    _ = get_run(session, run_id)

    stmt = select(PredictionResult).where(
        PredictionResult.run_id == run_id
    )

    if top_only:
        stmt = stmt.where(
            PredictionResult.is_top_k.is_(True)
        )

    stmt = stmt.order_by(
        PredictionResult.risk_rank.asc().nullslast(),
        PredictionResult.id.asc(),
    ).limit(limit)

    return list(session.execute(stmt).scalars().all())