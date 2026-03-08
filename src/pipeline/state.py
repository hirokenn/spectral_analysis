from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from src.data.dataset import Dataset
from src.model.base_model import BaseModel, ModelType


@dataclass
class BaseState:
    """
    Train / Predict 共通の状態。
    入出力をここに乗せていく。
    """

    dataset: Dataset
    model_type: ModelType
    model: BaseModel
    model_uri: str | None = None


@dataclass
class TrainState(BaseState):
    """
    学習時専用の情報を追加した State。
    """

    pass


@dataclass
class PredictState(BaseState):
    """
    予測時専用の情報を追加した State。
    """

    pass


StateLike = Union[TrainState, PredictState]
