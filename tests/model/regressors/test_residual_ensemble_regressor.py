from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import Dataset
from src.model.base_model import ModelType
from src.model.regressors.residual_ensemble import ResidualEnsembleRegressor
from src.preprocess.core.base import ComposablePreprocessor
from tests.helpers import make_dataset


@dataclass
class MultiplyPreprocessor(ComposablePreprocessor):
    """特徴量を定数倍するテスト用前処理。"""

    factor: float

    def fit(self, ds: Dataset) -> None:
        """学習対象を持たないため何もしない。"""
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        """特徴量を定数倍した Dataset を返す。"""
        return ds.with_features(ds.X * self.factor)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """特徴量を定数倍した Dataset を返す。"""
        return self.transform(ds)


@dataclass
class FirstColumnModel:
    """1 列目をそのまま予測値として返すテスト用モデル。"""

    fitted_X: np.ndarray | None = None

    @property
    def model_type(self) -> ModelType:
        return ModelType.SKLEARN

    def fit(self, ds: Dataset) -> None:
        """学習時入力を記録する。"""
        self.fitted_X = ds.X.copy()

    def predict(self, ds: Dataset) -> np.ndarray:
        """1 列目を予測値として返す。"""
        return ds.X[:, 0].astype(np.float32)

    def get_native_model(self) -> Any:
        """保存用に自身を返す。"""
        return self

    def set_native_model(self, native_model: Any) -> None:
        """保存済みモデルから状態を取り込む。"""
        self.fitted_X = native_model.fitted_X


@dataclass
class MeanResidualModel:
    """残差ターゲット平均を返すテスト用モデル。"""

    fitted_X: np.ndarray | None = None
    fitted_y: np.ndarray | None = None
    mean_: float = 0.0

    @property
    def model_type(self) -> ModelType:
        return ModelType.SKLEARN

    def fit(self, ds: Dataset) -> None:
        """学習時入力を記録する。"""
        assert ds.y is not None
        self.fitted_X = ds.X.copy()
        self.fitted_y = ds.y.copy()
        self.mean_ = float(np.mean(ds.y))

    def predict(self, ds: Dataset) -> np.ndarray:
        """学習した平均残差を返す。"""
        return np.full(ds.X.shape[0], self.mean_, dtype=np.float32)

    def get_native_model(self) -> Any:
        """保存用に自身を返す。"""
        return self

    def set_native_model(self, native_model: Any) -> None:
        """保存済みモデルから状態を取り込む。"""
        self.fitted_X = native_model.fitted_X
        self.fitted_y = native_model.fitted_y
        self.mean_ = float(native_model.mean_)


def test_residual_ensemble_uses_separate_preprocessors_and_residual_target() -> None:
    train_ds = make_dataset(
        X=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        y=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        sample_id=np.array([1, 2, 3]),
    )
    test_ds = make_dataset(
        X=np.array([[4.0], [5.0]], dtype=np.float32),
        y=None,
        sample_id=np.array([10, 11]),
    )
    base_model = FirstColumnModel()
    residual_model = MeanResidualModel()
    model = ResidualEnsembleRegressor(
        base_preprocessor=MultiplyPreprocessor(factor=2.0),
        base_model=base_model,
        residual_preprocessor=MultiplyPreprocessor(factor=3.0),
        residual_model=residual_model,
    )

    model.fit(train_ds)
    predictions = model.predict(test_ds)

    assert base_model.fitted_X is not None
    assert residual_model.fitted_X is not None
    assert residual_model.fitted_y is not None
    np.testing.assert_allclose(
        base_model.fitted_X,
        np.array([[2.0], [4.0], [6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        residual_model.fitted_X,
        np.array([[3.0], [6.0], [9.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        residual_model.fitted_y,
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        predictions,
        np.array([8.0, 10.0], dtype=np.float32),
    )
