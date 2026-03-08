from __future__ import annotations

from dataclasses import dataclass, field

from src.data.dataset import Dataset
from src.preprocess.base import BasePreprocessor


@dataclass
class PreprocessingPipeline:
    """複数前処理を順に適用するパイプライン。"""

    steps: list[BasePreprocessor] = field(default_factory=list)

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
