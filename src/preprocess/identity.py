from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.dataset import Dataset
from src.preprocess.base import BasePreprocessor

if TYPE_CHECKING:
    from src.preprocess.pipeline import PreprocessingPipeline


@dataclass
class IdentityPreprocessor:
    """入力データをそのまま返す no-op 前処理。"""

    def __rshift__(
        self, other: BasePreprocessor | PreprocessingPipeline
    ) -> PreprocessingPipeline:
        """前処理を `>>` で連結してパイプライン化する。"""
        from src.preprocess.pipeline import PreprocessingPipeline

        return PreprocessingPipeline() >> self >> other

    def fit(self, ds: Dataset) -> None:
        """学習対象を持たないため何もしない。"""
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        """入力をそのまま返す。"""
        return ds

    def fit_transform(self, ds: Dataset) -> Dataset:
        """入力をそのまま返す。"""
        return ds
