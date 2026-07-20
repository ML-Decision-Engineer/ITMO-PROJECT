from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings


def build_engine():
    connect_args = {}
    poolclass = None

    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if settings.database_url.endswith(":memory:"):
            poolclass = StaticPool

    kwargs = {
        "pool_pre_ping": True,
        "future": True,
        "connect_args": connect_args,
    }
    if poolclass is not None:
        kwargs["poolclass"] = poolclass

    return create_engine(settings.database_url, **kwargs)


engine = build_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def database_is_available() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
