from __future__ import annotations

import numpy as np

from src.model.base_model import ModelType
from src.model.regressors.lightgbm import LightGBMRegressor
from tests.helpers import make_dataset


def make_regression_dataset(with_target: bool = True):
    """LightGBM テスト用の小さな `Dataset` を作成する。"""
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32) if with_target else None
    return make_dataset(
        X=X,
        y=y,
        sample_id=np.array([1, 2, 3, 4], dtype=np.int64),
        groups=np.array([1, 1, 2, 2], dtype=np.int64),
        feature_names=["f0", "f1"],
    )


def test_lightgbm_regressor_fits_and_predicts() -> None:
    ds = make_regression_dataset()
    model = LightGBMRegressor(n_estimators=20, min_child_samples=1)

    model.fit(ds)
    pred = model.predict(ds)

    assert model.model_type == ModelType.LIGHTGBM
    assert pred.shape == (4,)


def test_lightgbm_regressor_raises_without_target() -> None:
    ds = make_regression_dataset(with_target=False)
    model = LightGBMRegressor(n_estimators=10)

    try:
        model.fit(ds)
    except ValueError as exc:
        assert "y" in str(exc)
    else:
        raise AssertionError("ValueError が送出されませんでした")


def test_lightgbm_regressor_returns_feature_importance() -> None:
    ds = make_regression_dataset()
    model = LightGBMRegressor(n_estimators=20, min_child_samples=1)
    model.fit(ds)

    importance = model.feature_importance()

    assert importance is not None
    assert set(importance) == {"f0", "f1"}
