from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.core.base import ComposablePreprocessor
from src.preprocess.core.feature_union import FeatureUnion
from src.preprocess.core.pipeline import PreprocessingPipeline
from src.preprocess.transforms.identity import IdentityPreprocessor
from tests.helpers import make_dataset


@dataclass
class MultiplyPreprocessor(ComposablePreprocessor):
    """入力特徴量を定数倍するテスト用前処理。"""

    factor: float

    def fit(self, ds: Dataset) -> None:
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        return ds.with_features(ds.X * self.factor)

    def fit_transform(self, ds: Dataset) -> Dataset:
        return self.transform(ds)


@dataclass
class RowMeanFeaturePreprocessor(ComposablePreprocessor):
    """行平均を追加特徴量として返すテスト用前処理。"""

    feature_name: str = "row_mean"

    def fit(self, ds: Dataset) -> None:
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        mean_feature = ds.X.mean(axis=1, keepdims=True)
        return ds.with_features(mean_feature, feature_names=[self.feature_name])

    def fit_transform(self, ds: Dataset) -> Dataset:
        return self.transform(ds)


def test_preprocessing_pipeline_supports_rshift_composition() -> None:
    pipeline = (
        PreprocessingPipeline() >> IdentityPreprocessor() >> IdentityPreprocessor()
    )

    assert isinstance(pipeline, PreprocessingPipeline)
    assert len(pipeline.steps) == 2


def test_identity_preprocessor_supports_rshift_composition() -> None:
    pipeline = IdentityPreprocessor() >> IdentityPreprocessor()
    ds = make_dataset(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        y=np.array([1.0], dtype=np.float32),
        sample_id=np.array([1], dtype=np.int64),
        groups=np.array([1], dtype=np.int64),
    )

    transformed = pipeline.fit_transform(ds)

    assert isinstance(pipeline, PreprocessingPipeline)
    np.testing.assert_allclose(transformed.X, ds.X)


def test_feature_union_concatenates_branch_outputs() -> None:
    pipeline = MultiplyPreprocessor(2.0) >> FeatureUnion(
        branches=[
            IdentityPreprocessor(),
            RowMeanFeaturePreprocessor("mean_after_scale"),
        ]
    )
    ds = make_dataset(
        X=np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32),
        y=np.array([1.0, 2.0], dtype=np.float32),
        sample_id=np.array([1, 2], dtype=np.int64),
        groups=np.array([1, 2], dtype=np.int64),
        feature_names=["f0", "f1"],
    )

    transformed = pipeline.fit_transform(ds)

    np.testing.assert_allclose(
        transformed.X,
        np.array([[2.0, 4.0, 3.0], [6.0, 10.0, 8.0]], dtype=np.float32),
    )
    assert transformed.feature_names is not None
    assert transformed.feature_names.tolist() == ["f0", "f1", "mean_after_scale"]
