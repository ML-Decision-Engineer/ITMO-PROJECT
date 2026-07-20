from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import database_is_available, get_db
from app.queue import publish_daily_task, rabbitmq_is_available
from app.runtime import available_days, get_bundle, get_dataset
from app.services import (
    NotFoundError,
    ServiceError,
    create_daily_run,
    create_single_prediction,
    get_model_metadata,
    get_run,
    init_database,
    list_results,
    list_runs,
    mark_run_failed,
    serialize_model_metadata,
    serialize_result,
    serialize_run,
)


logger = logging.getLogger(__name__)


class DailyPredictionRequest(BaseModel):
    production_day: date = Field(
        description="Производственный день в формате YYYY-MM-DD",
        examples=["2026-04-10"],
    )

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


app = FastAPI(
    title="MVP прогнозирования дефектов продукции",
    version="1.0.0",
    description=(
        "REST API для одиночного прогноза риска дефекта и асинхронного "
        "суточного расчёта через RabbitMQ и масштабируемые ML-воркеры."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(NotFoundError)
async def not_found_handler(_, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ServiceError)
async def service_error_handler(_, exc: ServiceError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health", tags=["system"])
def health():
    model_ok = True
    data_ok = True

    try:
        _ = get_bundle()
    except Exception:
        model_ok = False

    try:
        _ = get_dataset()
    except Exception:
        data_ok = False

    database_ok = database_is_available()
    rabbitmq_ok = rabbitmq_is_available()

    all_components_ok = (
        model_ok
        and data_ok
        and database_ok
        and rabbitmq_ok
    )

    return {
        "status": "ok" if all_components_ok else "degraded",
        "model": model_ok,
        "data": data_ok,
        "database": database_ok,
        "rabbitmq": rabbitmq_ok,
    }


@app.get("/api/model", tags=["model"])
def model_info(db: Session = Depends(get_db)):
    return serialize_model_metadata(get_model_metadata(db))


@app.get("/api/data/dates", tags=["data"])
def data_dates():
    dates = available_days()
    return {"dates": dates, "total_days": len(dates)}


@app.post("/api/predictions/random-row", tags=["predictions"])
def random_row_prediction(db: Session = Depends(get_db)):
    run, result = create_single_prediction(db)
    return {
        "run": serialize_run(run),
        "result": serialize_result(result),
    }


@app.post(
    "/api/predictions/day",
    tags=["predictions"],
    status_code=status.HTTP_202_ACCEPTED,
)
def daily_prediction(payload: DailyPredictionRequest, db: Session = Depends(get_db)):
    run = create_daily_run(
        db,
        payload.production_day.isoformat(),
    )
    try:
        publish_daily_task(run.id)
    except Exception as exc:
        mark_run_failed(db, run, exc)
        raise HTTPException(status_code=503, detail="Не удалось поставить задачу в очередь") from exc

    return serialize_run(run)


@app.get("/api/predictions/runs", tags=["history"])
def prediction_runs(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return {"runs": [serialize_run(item) for item in list_runs(db, limit)]}


@app.get("/api/predictions/runs/{run_id}", tags=["history"])
def prediction_run(run_id: int, db: Session = Depends(get_db)):
    return serialize_run(get_run(db, run_id))


@app.get("/api/predictions/runs/{run_id}/results", tags=["history"])
def prediction_results(
    run_id: int,
    top_only: bool = False,
    limit: int = Query(default=5000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    items = list_results(db, run_id=run_id, top_only=top_only, limit=limit)
    return {
        "run_id": run_id,
        "top_only": top_only,
        "results": [serialize_result(item) for item in items],
    }
