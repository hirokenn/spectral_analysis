from __future__ import annotations

from dataclasses import dataclass, field

from src.data.dataset import Dataset
from src.preprocess.core.base import BasePreprocessor, ComposablePreprocessor


@dataclass
class FeatureUnion(ComposablePreprocessor):
    """複数 branch の出力特徴量を横方向に結合する前処理。

    各 branch には同じ入力 `Dataset` が渡され、branch ごとの出力 `X` を
    列方向に連結した新しい `Dataset` を返す。
    """

    branches: list[BasePreprocessor] = field(default_factory=list)

    def __post_init__(self) -> None:
        """少なくとも 1 つ以上の branch があることを検証する。"""
        if not self.branches:
            raise ValueError("FeatureUnion には 1 つ以上の branch が必要です")

    def fit(self, ds: Dataset) -> None:
        """全 branch を同じ入力データで学習する。"""
        for branch in self.branches:
            branch.fit(ds)

    def transform(self, ds: Dataset) -> Dataset:
        """全 branch の出力を横結合した `Dataset` を返す。"""
        transformed = [branch.transform(ds) for branch in self.branches]
        return Dataset.concat_features(transformed)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """全 branch を学習しながら適用し、出力を横結合する。"""
        transformed = [branch.fit_transform(ds) for branch in self.branches]
        return Dataset.concat_features(transformed)
