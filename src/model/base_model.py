from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.data.dataset import Dataset
    import numpy as np


class ModelType(StrEnum):
    """サポートするモデル実装タイプ。"""

    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"


@runtime_checkable
class BaseModel(Protocol):
    """全モデル共通のインターフェース。"""

    @property
    def model_type(self) -> ModelType: ...

    def fit(self, ds: Dataset) -> None: ...

    def predict(self, ds: Dataset) -> np.ndarray: ...

    def get_native_model(self) -> Any:
        """保存対象のネイティブモデルオブジェクトを返す。"""
        ...

    def set_native_model(self, native_model: Any) -> None:
        """読み込んだネイティブモデルを設定し、学習済み状態にする。"""
        ...
