from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.data.dataset import Dataset
    import numpy as np


@runtime_checkable
class BaseModel(Protocol):
    def fit(self, ds: Dataset) -> None: ...

    def predict(self, ds: Dataset) -> np.ndarray: ...
