from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://homework8:homework8@database:5432/homework8",
    )

    rabbitmq_host: str = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbitmq_user: str = os.getenv("RABBITMQ_USER", "homework8")
    rabbitmq_password: str = os.getenv("RABBITMQ_PASSWORD", "homework8")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "daily_prediction_queue")

    bundle_path: Path = Path(
        os.getenv("MODEL_BUNDLE_PATH", "/project/artifacts/model_bundle.joblib")
    )
    data_path: Path = Path(os.getenv("DATA_PATH", "/project/data_mvp.csv"))


settings = Settings()
