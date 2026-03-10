from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from src.data.dataset import Dataset
from src.model.base_model import BaseModel, ModelType

if TYPE_CHECKING:
    from src.preprocess.core.base import BasePreprocessor


@dataclass
class BaseState:
    """
    Train / Predict 共通の状態。
    入出力をここに乗せていく。
    """

    dataset: Dataset | None = None
    train_dataset: Dataset | None = None
    test_dataset: Dataset | None = None
    recipe_name: str | None = None
    model_type: ModelType | None = None
    model: BaseModel | None = None
    preprocessor: BasePreprocessor | None = None
    model_uri: str | None = None
    run_id: str | None = None
    test_predictions: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainState(BaseState):
    """
    学習時専用の情報を追加した State。
    """

    oof_predictions: np.ndarray | None = None

    pass


@dataclass
class PredictState(BaseState):
    """
    予測時専用の情報を追加した State。
    """

    pass


StateLike = Union[TrainState, PredictState]
