from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import Dataset
from src.model.regressors.pls import PLSRegressor


def make_dataset(with_target: bool = True) -> Dataset:
    """PLS テスト用の小さな `Dataset` を作成する。"""
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) if with_target else None
    return Dataset(
        X=X,
        y=y,
        groups=np.array([1, 1, 2, 2], dtype=np.int64),
        sample_id=np.array([10, 11, 12, 13], dtype=np.int64),
        wavenumbers=np.array([10000.0, 9990.0, 9980.0], dtype=np.float32),
    )


def test_pls_fit_and_predict_returns_1d_array() -> None:
    model = PLSRegressor(n_components=2)
    ds = make_dataset(with_target=True)

    model.fit(ds)
    pred = model.predict(ds)

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_pls_fit_raises_when_target_is_missing() -> None:
    model = PLSRegressor(n_components=2)
    ds = make_dataset(with_target=False)

    with pytest.raises(ValueError, match="y"):
        model.fit(ds)


def test_pls_predict_raises_before_fit() -> None:
    model = PLSRegressor(n_components=2)
    ds = make_dataset(with_target=True)

    with pytest.raises(ValueError, match="未学習"):
        model.predict(ds)
