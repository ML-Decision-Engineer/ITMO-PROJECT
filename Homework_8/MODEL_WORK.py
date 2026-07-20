from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_bundle(
    bundle_path: str | Path = "artifacts/model_bundle.joblib",
) -> dict[str, Any]:
    """Загрузить и проверить ML bundle."""

    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Не найден model bundle: {bundle_path}"
        )

    bundle = joblib.load(bundle_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Model bundle должен быть словарём."
        )

    required_keys = {
        "model",
        "feature_names",
        "categorical_features",
        "numeric_features",
        "feature_count",
        "threshold_raw",
        "na_token",
    }

    missing_keys = sorted(
        required_keys - set(bundle)
    )

    if missing_keys:
        raise ValueError(
            "В model bundle отсутствуют поля: "
            + ", ".join(missing_keys)
        )

    feature_names = list(
        bundle["feature_names"]
    )
    categorical_features = list(
        bundle["categorical_features"]
    )
    numeric_features = list(
        bundle["numeric_features"]
    )

    if not feature_names:
        raise ValueError(
            "Список признаков модели пуст."
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "В feature_names присутствуют дубликаты."
        )

    expected_feature_count = int(
        bundle["feature_count"]
    )

    if len(feature_names) != expected_feature_count:
        raise ValueError(
            "feature_count не совпадает с длиной feature_names: "
            f"{expected_feature_count} != {len(feature_names)}"
        )

    categorical_set = set(
        categorical_features
    )
    numeric_set = set(
        numeric_features
    )
    feature_set = set(
        feature_names
    )

    overlap = sorted(
        categorical_set & numeric_set
    )

    if overlap:
        raise ValueError(
            "Признаки одновременно указаны как числовые "
            f"и категориальные: {overlap[:10]}"
        )

    unknown_typed_features = sorted(
        (categorical_set | numeric_set)
        - feature_set
    )

    if unknown_typed_features:
        raise ValueError(
            "В списках типов присутствуют признаки, "
            "которых нет в feature_names: "
            f"{unknown_typed_features[:10]}"
        )

    untyped_features = sorted(
        feature_set
        - categorical_set
        - numeric_set
    )

    if untyped_features:
        raise ValueError(
            "Для части признаков не указан тип: "
            f"{untyped_features[:10]}"
        )

    target_col = bundle.get(
        "target_col",
        "actual_target",
    )

    if target_col in feature_set:
        raise ValueError(
            f"Таргет `{target_col}` входит в признаки модели."
        )

    technical_columns = {
        bundle.get("id_col"),
        bundle.get("time_col"),
        bundle.get("day_col"),
    }
    technical_columns.discard(None)

    leaked_technical_columns = sorted(
        feature_set & technical_columns
    )

    if leaked_technical_columns:
        raise ValueError(
            "Технические колонки попали в признаки модели: "
            f"{leaked_technical_columns}"
        )

    threshold_raw = float(
        bundle["threshold_raw"]
    )

    if not 0.0 <= threshold_raw <= 1.0:
        raise ValueError(
            "threshold_raw должен находиться в диапазоне [0, 1]."
        )

    top_k_share = float(
        bundle.get("top_k_share", 0.10)
    )

    if not 0.0 < top_k_share <= 1.0:
        raise ValueError(
            "top_k_share должен находиться в диапазоне (0, 1]."
        )

    model = bundle["model"]

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Сохранённая модель не поддерживает predict_proba()."
        )

    calibrator = bundle.get("calibrator")

    if (
        calibrator is not None
        and not hasattr(calibrator, "predict")
    ):
        raise TypeError(
            "Сохранённый калибратор не поддерживает predict()."
        )

    return bundle


def load_data(
    data_path: str | Path,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Загрузить и проверить демонстрационный CSV."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден датасет: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            f"Датасет пустой: {data_path}"
        )

    id_col = bundle.get(
        "id_col",
        "row_id",
    )
    time_col = bundle.get(
        "time_col",
        "production_time",
    )
    day_col = bundle.get(
        "day_col",
        "production_day",
    )
    target_col = bundle.get(
        "target_col",
        "actual_target",
    )

    feature_names = list(
        bundle["feature_names"]
    )

    required_columns = [
        id_col,
        target_col,
        *feature_names,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "В демонстрационном датасете отсутствуют "
            "обязательные колонки: "
            f"{missing_columns[:20]}. "
            f"Всего отсутствует: {len(missing_columns)}"
        )

    if (
        day_col not in df.columns
        and time_col not in df.columns
    ):
        raise ValueError(
            f"В данных нет ни `{day_col}`, ни `{time_col}`."
        )

    if time_col in df.columns:
        parsed_time = pd.to_datetime(
            df[time_col],
            errors="coerce",
        )

        invalid_time_count = int(
            parsed_time.isna().sum()
        )

        if invalid_time_count > 0:
            raise ValueError(
                f"В колонке `{time_col}` найдено "
                f"некорректных или пустых значений: "
                f"{invalid_time_count}"
            )

        df[time_col] = parsed_time

    if day_col in df.columns:
        parsed_day = pd.to_datetime(
            df[day_col],
            errors="coerce",
        )

        invalid_day_count = int(
            parsed_day.isna().sum()
        )

        if invalid_day_count > 0:
            raise ValueError(
                f"В колонке `{day_col}` найдено "
                f"некорректных или пустых значений: "
                f"{invalid_day_count}"
            )

        df[day_col] = parsed_day.dt.strftime(
            "%Y-%m-%d"
        )

    else:
        df[day_col] = df[
            time_col
        ].dt.strftime("%Y-%m-%d")

    if df[id_col].isna().any():
        raise ValueError(
            f"В колонке `{id_col}` присутствуют пустые ID."
        )

    df[id_col] = df[id_col].astype(str)

    target_numeric = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    invalid_target_count = int(
        target_numeric.isna().sum()
    )

    if invalid_target_count > 0:
        raise ValueError(
            f"В колонке `{target_col}` найдено "
            f"некорректных или пустых значений: "
            f"{invalid_target_count}"
        )

    unique_targets = set(
        target_numeric.unique().tolist()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Таргет `{target_col}` должен быть бинарным 0/1. "
            f"Найдены значения: {sorted(unique_targets)[:20]}"
        )

    df[target_col] = target_numeric.astype(
        "int8"
    )

    return df


def prepare_features(
    df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """
    Проверить схему и подготовить признаки
    в точном порядке обучения модели.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "На вход ожидается pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Нельзя выполнить прогноз для пустого DataFrame."
        )

    feature_names = list(
        bundle["feature_names"]
    )

    numeric_features = list(
        bundle["numeric_features"]
    )

    categorical_features = list(
        bundle["categorical_features"]
    )

    na_token = str(
        bundle["na_token"]
    )

    missing_features = [
        column
        for column in feature_names
        if column not in df.columns
    ]

    if missing_features:
        preview = missing_features[:20]

        raise ValueError(
            "В данных отсутствуют обязательные признаки: "
            f"{preview}. "
            f"Всего отсутствует: {len(missing_features)}"
        )

    # Выбираю только признаки модели.
    # Таргет, дата, ID и дополнительные поля сюда не попадут.
    X = df.loc[
        :,
        feature_names,
    ].copy()

    X = X.mask(
        X.isin([np.inf, -np.inf]),
        np.nan,
    )

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype("object")
            .where(
                X[column].notna(),
                na_token,
            )
            .astype(str)
        )

    if list(X.columns) != feature_names:
        raise AssertionError(
            "Нарушен порядок признаков перед моделью."
        )

    if X.shape[1] != int(
        bundle["feature_count"]
    ):
        raise AssertionError(
            "Число признаков не совпадает "
            "с числом признаков модели."
        )

    return X


def predict_rows(
    df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Получить прогноз для одной или нескольких строк."""

    X = prepare_features(
        df,
        bundle,
    )

    model = bundle["model"]
    calibrator = bundle.get("calibrator")

    raw_score = np.asarray(
        model.predict_proba(X)[:, 1],
        dtype=float,
    )

    if calibrator is not None:
        calibrated_probability = np.asarray(
            calibrator.predict(raw_score),
            dtype=float,
        )
    else:
        calibrated_probability = raw_score.copy()

    calibrated_probability = np.clip(
        calibrated_probability,
        0.0,
        1.0,
    )

    threshold_raw = float(
        bundle["threshold_raw"]
    )

    predicted_class = (
        raw_score >= threshold_raw
    ).astype("int8")

    meta_columns = [
        bundle.get("id_col"),
        bundle.get("time_col"),
        bundle.get("day_col"),
        bundle.get("target_col"),
    ]

    meta_columns = [
        column
        for column in meta_columns
        if column and column in df.columns
    ]

    result = (
        df[meta_columns]
        .reset_index(drop=True)
        .copy()
    )

    result["raw_score"] = raw_score

    # Это значение используем как прогнозную вероятность
    # при расчёте суточного процента.
    result["defect_probability"] = (
        calibrated_probability
    )

    # Бинарное решение строится по raw-порогу,
    # потому что именно так порог оценивался в notebook.
    result["predicted_class"] = (
        predicted_class
    )

    result["risk"] = np.where(
        predicted_class == 1,
        "Высокий",
        "Обычный",
    )

    target_col = bundle.get(
        "target_col",
        "actual_target",
    )

    if target_col in result.columns:
        actual = pd.to_numeric(
            result[target_col],
            errors="coerce",
        )

        result["prediction_correct"] = np.where(
            actual.isna(),
            pd.NA,
            (
                actual.astype("Int64")
                == result["predicted_class"]
            ),
        )

    return result


def predict_random_row(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    random_state: int | None = None,
) -> pd.DataFrame:
    """Выбрать случайную строку и выполнить прогноз."""

    if data.empty:
        raise ValueError(
            "Датасет пустой."
        )

    row = data.sample(
        n=1,
        random_state=random_state,
    )

    return predict_rows(
        row,
        bundle,
    )


def predict_day(
    data: pd.DataFrame,
    production_day: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Выполнить расчёт для всех строк выбранного дня."""

    day_col = bundle.get(
        "day_col",
        "production_day",
    )

    target_col = bundle.get(
        "target_col",
        "actual_target",
    )

    requested_day = (
        pd.Timestamp(production_day)
        .strftime("%Y-%m-%d")
    )

    day_data = data[
        data[day_col].astype(str)
        == requested_day
    ].copy()

    if day_data.empty:
        raise ValueError(
            f"За день {requested_day} строки не найдены."
        )

    result = predict_rows(
        day_data,
        bundle,
    )

    result = (
        result
        .sort_values(
            "defect_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["risk_rank"] = (
        np.arange(len(result)) + 1
    )

    top_k_share = float(
        bundle.get(
            "top_k_share",
            0.10,
        )
    )

    top_k_count = max(
        1,
        int(
            np.ceil(
                len(result) * top_k_share
            )
        ),
    )

    result["is_top_k"] = (
        result["risk_rank"]
        <= top_k_count
    )

    predicted_daily_rate = float(
        result["defect_probability"].mean()
        * 100
    )

    flagged_count = int(
        result["predicted_class"].sum()
    )

    flagged_share = float(
        result["predicted_class"].mean()
    )

    actual_daily_rate = None
    absolute_error_pp = None

    if (
        target_col in result.columns
        and result[target_col].notna().any()
    ):
        actual_daily_rate = float(
            pd.to_numeric(
                result[target_col],
                errors="coerce",
            ).mean()
            * 100
        )

        absolute_error_pp = abs(
            predicted_daily_rate
            - actual_daily_rate
        )

    summary = {
        "production_day": requested_day,
        "row_count": int(len(result)),
        "predicted_daily_rate": (
            predicted_daily_rate
        ),
        "actual_daily_rate": (
            actual_daily_rate
        ),
        "absolute_error_pp": (
            absolute_error_pp
        ),
        "flagged_count": flagged_count,
        "flagged_share": flagged_share,
        "top_k_share": top_k_share,
        "top_k_count": top_k_count,
    }

    return {
        "summary": summary,
        "results": result,
        "top_k": result[
            result["is_top_k"]
        ].copy(),
    }
