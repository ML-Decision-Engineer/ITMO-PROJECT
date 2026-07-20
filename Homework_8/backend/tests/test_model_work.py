from __future__ import annotations

import math

import numpy as np
import pytest

from MODEL_WORK import (
    predict_day,
    predict_rows,
    prepare_features,
)

from app.runtime import (
    get_bundle,
    get_dataset,
)


def test_bundle_and_dataset_load():
    bundle = get_bundle()
    data = get_dataset()

    assert not data.empty

    assert (
        bundle["feature_count"]
        == len(bundle["feature_names"])
    )

    assert (
        set(bundle["categorical_features"])
        | set(bundle["numeric_features"])
        == set(bundle["feature_names"])
    )

    target_col = bundle.get(
        "target_col",
        "actual_target",
    )
    day_col = bundle.get(
        "day_col",
        "production_day",
    )
    id_col = bundle.get(
        "id_col",
        "row_id",
    )

    assert target_col in data.columns
    assert day_col in data.columns
    assert id_col in data.columns

    assert data[target_col].notna().all()
    assert data[day_col].notna().all()
    assert data[id_col].notna().all()

    assert set(
        data[target_col].unique()
    ).issubset({0, 1})


def test_preprocessing_uses_exact_feature_order_and_not_target():
    bundle = get_bundle()
    data = get_dataset().head(5)

    prepared = prepare_features(
        data,
        bundle,
    )

    assert list(
        prepared.columns
    ) == bundle["feature_names"]

    assert (
        prepared.shape[1]
        == bundle["feature_count"]
    )

    assert bundle.get(
        "target_col",
        "actual_target",
    ) not in prepared.columns


def test_missing_feature_raises_error():
    bundle = get_bundle()
    data = get_dataset().head(3).copy()

    missing_feature = bundle[
        "feature_names"
    ][0]

    broken_data = data.drop(
        columns=[missing_feature]
    )

    with pytest.raises(
        ValueError,
        match="отсутствуют обязательные признаки",
    ):
        prepare_features(
            broken_data,
            bundle,
        )


def test_extra_columns_are_ignored():
    bundle = get_bundle()
    data = get_dataset().head(3).copy()

    data["unused_extra_column"] = 123

    prepared = prepare_features(
        data,
        bundle,
    )

    assert (
        "unused_extra_column"
        not in prepared.columns
    )

    assert list(
        prepared.columns
    ) == bundle["feature_names"]


def test_row_predictions_have_valid_probability():
    bundle = get_bundle()

    result = predict_rows(
        get_dataset().head(10),
        bundle,
    )

    assert len(result) == 10

    assert result[
        "raw_score"
    ].between(0, 1).all()

    assert result[
        "defect_probability"
    ].between(0, 1).all()

    assert result[
        "predicted_class"
    ].isin([0, 1]).all()


def test_binary_class_uses_raw_threshold():
    bundle = get_bundle()

    result = predict_rows(
        get_dataset().head(50),
        bundle,
    )

    expected_class = (
        result["raw_score"]
        >= float(bundle["threshold_raw"])
    ).astype("int8")

    assert np.array_equal(
        result["predicted_class"].to_numpy(),
        expected_class.to_numpy(),
    )


def test_daily_aggregates_and_top_k():
    bundle = get_bundle()
    data = get_dataset()

    day_col = bundle.get(
        "day_col",
        "production_day",
    )

    first_day = str(
        data[day_col].dropna().iloc[0]
    )

    output = predict_day(
        data,
        first_day,
        bundle,
    )

    results = output["results"]
    summary = output["summary"]

    expected_daily_rate = float(
        results["defect_probability"].mean()
        * 100
    )

    expected_flagged_count = int(
        results["predicted_class"].sum()
    )

    expected_flagged_share = float(
        results["predicted_class"].mean()
    )

    expected_top_k_count = max(
        1,
        math.ceil(
            len(results)
            * float(bundle.get("top_k_share", 0.10))
        ),
    )

    assert summary["row_count"] == len(
        results
    )

    assert summary[
        "predicted_daily_rate"
    ] == pytest.approx(
        expected_daily_rate
    )

    assert summary[
        "flagged_count"
    ] == expected_flagged_count

    assert summary[
        "flagged_share"
    ] == pytest.approx(
        expected_flagged_share
    )

    assert summary[
        "top_k_count"
    ] == expected_top_k_count

    assert int(
        results["is_top_k"].sum()
    ) == expected_top_k_count