from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from src.data.dataset import Dataset

if TYPE_CHECKING:
    from src.preprocess.pipeline import PreprocessingPipeline


@runtime_checkable
class BasePreprocessor(Protocol):
    """全前処理共通のインターフェース。"""

    def fit(self, ds: Dataset) -> None: ...

    def transform(self, ds: Dataset) -> Dataset: ...

    def fit_transform(self, ds: Dataset) -> Dataset: ...


class ComposablePreprocessor:
    """`>>` で前処理パイプラインを組み立てるための mixin。"""

    def __rshift__(
        self, other: BasePreprocessor | "PreprocessingPipeline"
    ) -> "PreprocessingPipeline":
        """右辺と連結した `PreprocessingPipeline` を返す。"""
        from src.preprocess.pipeline import PreprocessingPipeline

        return PreprocessingPipeline() >> cast(BasePreprocessor, self) >> other
