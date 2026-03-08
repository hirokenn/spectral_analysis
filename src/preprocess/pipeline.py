from __future__ import annotations

from dataclasses import dataclass, field

from src.data.dataset import Dataset
from src.preprocess.base import BasePreprocessor


@dataclass
class PreprocessingPipeline:
    """複数前処理を順に適用するパイプライン。"""

    steps: list[BasePreprocessor] = field(default_factory=list)

    def __rshift__(
        self, other: BasePreprocessor | "PreprocessingPipeline"
    ) -> "PreprocessingPipeline":
        """前処理を `>>` で連結した新しいパイプラインを返す。"""
        if isinstance(other, PreprocessingPipeline):
            return PreprocessingPipeline([*self.steps, *other.steps])
        return PreprocessingPipeline([*self.steps, other])

    def __rrshift__(
        self, other: BasePreprocessor | "PreprocessingPipeline"
    ) -> "PreprocessingPipeline":
        """左辺が前処理の場合の `>>` を受ける。"""
        if isinstance(other, PreprocessingPipeline):
            return PreprocessingPipeline([*other.steps, *self.steps])
        return PreprocessingPipeline([other, *self.steps])

    def fit(self, ds: Dataset) -> None:
        """各前処理を順に学習する。"""
        current = ds
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)

    def transform(self, ds: Dataset) -> Dataset:
        """各前処理を順に適用する。"""
        current = ds
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, ds: Dataset) -> Dataset:
        """各前処理を順に学習しながら適用する。"""
        current = ds
        for step in self.steps:
            current = step.fit_transform(current)
        return current
