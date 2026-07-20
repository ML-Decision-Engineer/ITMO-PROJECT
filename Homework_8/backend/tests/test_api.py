from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def test_health_model_and_dates(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "rabbitmq_is_available",
        lambda: True,
    )

    with TestClient(app) as client:
        health = client.get(
            "/api/health"
        )

        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model"] is True
        assert health.json()["data"] is True
        assert health.json()["database"] is True
        assert health.json()["rabbitmq"] is True

        model = client.get(
            "/api/model"
        )

        assert model.status_code == 200
        assert model.json()["feature_count"] > 0
        assert (
            model.json()["threshold_type"]
            == "raw_score"
        )
        assert (
            model.json()["probability_type"]
            == "calibrated_probability"
        )

        dates = client.get(
            "/api/data/dates"
        )

        assert dates.status_code == 200
        assert dates.json()["total_days"] > 0


def test_health_is_degraded_without_rabbitmq(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "rabbitmq_is_available",
        lambda: False,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/health"
        )

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["rabbitmq"] is False


def test_single_prediction_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/predictions/random-row"
        )

        assert response.status_code == 200

        body = response.json()
        run = body["run"]
        result = body["result"]

        assert run["status"] == "completed"

        assert (
            run["production_day"]
            == result["production_day"]
        )

        assert run["row_count"] == 1

        assert abs(
            run["predicted_defect_rate"]
            - result["probability"] * 100
        ) < 1e-12

        assert run["top_k_count"] == 1

        assert (
            0
            <= result["probability"]
            <= 1
        )

        assert result[
            "predicted_class"
        ] in [0, 1]

        assert result["risk_rank"] == 1
        assert result["is_top_k"] is True

        stored = client.get(
            f"/api/predictions/runs/"
            f"{run['id']}/results"
        )

        assert stored.status_code == 200
        assert len(
            stored.json()["results"]
        ) == 1

        assert (
            stored.json()["results"][0]["row_id"]
            == result["row_id"]
        )

        history = client.get(
            "/api/predictions/runs"
        )

        assert history.status_code == 200
        assert len(
            history.json()["runs"]
        ) >= 1


def test_daily_endpoint_creates_queued_task(
    monkeypatch,
):
    published = []

    monkeypatch.setattr(
        main_module,
        "publish_daily_task",
        lambda run_id: published.append(run_id),
    )

    with TestClient(app) as client:
        first_day = client.get(
            "/api/data/dates"
        ).json()["dates"][0]["production_day"]

        response = client.post(
            "/api/predictions/day",
            json={
                "production_day": first_day,
            },
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"

    assert published == [
        response.json()["id"]
    ]


def test_invalid_date_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/api/predictions/day",
            json={
                "production_day": "not-a-date",
            },
        )

    assert response.status_code == 422


def test_unknown_day_returns_404():
    with TestClient(app) as client:
        response = client.post(
            "/api/predictions/day",
            json={
                "production_day": "2099-01-01",
            },
        )

    assert response.status_code == 404
    assert "строки не найдены" in response.json()[
        "detail"
    ]


def test_queue_error_marks_run_failed(
    monkeypatch,
):
    def raise_queue_error(_: int) -> None:
        raise RuntimeError(
            "forced queue error"
        )

    monkeypatch.setattr(
        main_module,
        "publish_daily_task",
        raise_queue_error,
    )

    with TestClient(app) as client:
        first_day = client.get(
            "/api/data/dates"
        ).json()["dates"][0]["production_day"]

        response = client.post(
            "/api/predictions/day",
            json={
                "production_day": first_day,
            },
        )

        assert response.status_code == 503

        history = client.get(
            "/api/predictions/runs",
            params={"limit": 1},
        ).json()["runs"]

    assert len(history) == 1
    assert history[0]["status"] == "failed"

    assert (
        "forced queue error"
        in history[0]["error_message"]
    )