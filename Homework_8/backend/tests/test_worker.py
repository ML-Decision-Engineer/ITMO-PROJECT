from __future__ import annotations

import pytest

import app.services as services_module
from app.db import SessionLocal
from app.runtime import available_days
from app.services import (
    create_daily_run,
    get_run,
    list_results,
    process_daily_run,
)


def test_worker_completes_daily_run_and_saves_results():
    first_day = available_days()[0][
        "production_day"
    ]

    with SessionLocal() as session:
        run = create_daily_run(
            session,
            first_day,
        )

        run_id = run.id
        expected_rows = run.row_count

    completed = process_daily_run(
        run_id,
        worker_id="pytest-worker",
    )

    assert completed.status == "completed"
    assert completed.worker_id == "pytest-worker"
    assert completed.row_count == expected_rows
    assert completed.started_at is not None
    assert completed.completed_at is not None

    assert (
        completed.predicted_defect_rate
        is not None
    )

    assert completed.flagged_count is not None
    assert completed.flagged_share is not None
    assert completed.top_k_count is not None

    with SessionLocal() as session:
        results = list_results(
            session,
            run_id,
        )

    assert len(results) == expected_rows

    assert sum(
        item.is_top_k
        for item in results
    ) == completed.top_k_count

    assert sum(
        item.predicted_class
        for item in results
    ) == completed.flagged_count

    assert (
        completed.flagged_share
        == pytest.approx(
            completed.flagged_count
            / completed.row_count
        )
    )


def test_worker_marks_run_failed(
    monkeypatch,
):
    first_day = available_days()[0][
        "production_day"
    ]

    with SessionLocal() as session:
        run = create_daily_run(
            session,
            first_day,
        )
        run_id = run.id

    def forced_prediction_error(*args, **kwargs):
        raise RuntimeError(
            "forced prediction error"
        )

    monkeypatch.setattr(
        services_module,
        "predict_day",
        forced_prediction_error,
    )

    with pytest.raises(
        RuntimeError,
        match="forced prediction error",
    ):
        process_daily_run(
            run_id,
            worker_id="pytest-worker",
        )

    with SessionLocal() as session:
        failed_run = get_run(
            session,
            run_id,
        )

        assert failed_run.status == "failed"
        assert failed_run.started_at is not None
        assert failed_run.completed_at is not None

        assert (
            "forced prediction error"
            in failed_run.error_message
        )