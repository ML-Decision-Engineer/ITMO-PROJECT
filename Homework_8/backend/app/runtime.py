from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from MODEL_WORK import load_bundle, load_data

from app.config import settings


@lru_cache(maxsize=1)
def get_bundle() -> dict[str, Any]:
    return load_bundle(settings.bundle_path)


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    return load_data(settings.data_path, get_bundle())


def available_days() -> list[dict[str, Any]]:
    bundle = get_bundle()
    data = get_dataset()
    day_col = bundle.get("day_col", "production_day")

    counts = (
        data.dropna(subset=[day_col])
        .groupby(day_col, observed=True)
        .size()
        .sort_index()
    )

    return [
        {"production_day": str(day), "row_count": int(count)}
        for day, count in counts.items()
    ]


def validate_runtime_files() -> None:
    missing = [
        str(path)
        for path in (settings.bundle_path, settings.data_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Не найдены обязательные файлы: " + ", ".join(missing))

    _ = get_bundle()
    _ = get_dataset()
