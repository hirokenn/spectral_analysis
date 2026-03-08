from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.data.dataset import Dataset


@runtime_checkable
class BasePreprocessor(Protocol):
    """全前処理共通のインターフェース。"""

    def fit(self, ds: Dataset) -> None: ...

    def transform(self, ds: Dataset) -> Dataset: ...

    def fit_transform(self, ds: Dataset) -> Dataset: ...
