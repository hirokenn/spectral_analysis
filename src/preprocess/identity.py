from __future__ import annotations

from dataclasses import dataclass

from src.data.dataset import Dataset
from src.preprocess.base import ComposablePreprocessor


@dataclass
class IdentityPreprocessor(ComposablePreprocessor):
    """入力データをそのまま返す no-op 前処理。"""

    def fit(self, ds: Dataset) -> None:
        """学習対象を持たないため何もしない。"""
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        """入力をそのまま返す。"""
        return ds

    def fit_transform(self, ds: Dataset) -> Dataset:
        """入力をそのまま返す。"""
        return ds
