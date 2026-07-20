from __future__ import annotations

import os
from pathlib import Path

import pytest


TEST_DB = Path("/tmp/homework8_test.db")
if TEST_DB.exists():
    TEST_DB.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ.setdefault("MODEL_BUNDLE_PATH", "/project/artifacts/model_bundle.joblib")
os.environ.setdefault("DATA_PATH", "/project/data_mvp.csv")
os.environ.setdefault("RABBITMQ_HOST", "rabbitmq")


@pytest.fixture(autouse=True)
def clean_database():
    from app.db import engine
    from app.models import Base
    from app.services import init_database

    Base.metadata.drop_all(bind=engine)
    init_database()
    yield
    Base.metadata.drop_all(bind=engine)
