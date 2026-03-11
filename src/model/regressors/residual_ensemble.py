from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import Dataset
from src.model.base_model import BaseModel, ModelType
from src.preprocess.core.base import BasePreprocessor


@dataclass
class ResidualEnsembleRegressor(BaseModel):
    """ベース予測と残差予測を加算する二段回帰器。"""

    base_preprocessor: BasePreprocessor
    base_model: BaseModel
    residual_preprocessor: BasePreprocessor
    residual_model: BaseModel

    @property
    def model_type(self) -> ModelType:
        """複合モデル全体を sklearn flavor として扱う。"""
        return ModelType.SKLEARN

    def fit(self, ds: Dataset) -> None:
        """ベースモデルを学習し、その残差で補正モデルを学習する。"""
        if ds.y is None:
            raise ValueError("学習には y が必要です")

        base_train_ds = self.base_preprocessor.fit_transform(ds)
        self.base_model.fit(base_train_ds)
        base_predictions = np.asarray(
            self.base_model.predict(base_train_ds),
            dtype=np.float32,
        )
        if base_predictions.shape != ds.y.shape:
            raise ValueError("ベースモデルの予測件数と y の件数が一致しません")

        residual_target = (ds.y - base_predictions).astype(np.float32)
        residual_train_ds = self.residual_preprocessor.fit_transform(ds).with_target(
            residual_target
        )
        self.residual_model.fit(residual_train_ds)

    def predict(self, ds: Dataset) -> np.ndarray:
        """ベース予測と残差予測の和を返す。"""
        base_predictions = np.asarray(
            self.base_model.predict(self.base_preprocessor.transform(ds)),
            dtype=np.float32,
        )
        residual_predictions = np.asarray(
            self.residual_model.predict(self.residual_preprocessor.transform(ds)),
            dtype=np.float32,
        )
        if base_predictions.shape != residual_predictions.shape:
            raise ValueError("ベース予測と残差予測の件数が一致しません")
        return (base_predictions + residual_predictions).astype(np.float32)

    def get_native_model(self) -> Any:
        """保存用に複合モデル自身を返す。"""
        return self

    def set_native_model(self, native_model: Any) -> None:
        """保存済みの複合モデルを取り込む。"""
        if not isinstance(native_model, ResidualEnsembleRegressor):
            raise ValueError("ResidualEnsembleRegressor を読み込めません")
        self.base_preprocessor = native_model.base_preprocessor
        self.base_model = native_model.base_model
        self.residual_preprocessor = native_model.residual_preprocessor
        self.residual_model = native_model.residual_model

    def feature_importance(self) -> dict[str, float] | None:
        """残差モデル側の特徴量重要度を返す。"""
        feature_importance_fn = getattr(self.residual_model, "feature_importance", None)
        if feature_importance_fn is None:
            return None
        importance = feature_importance_fn()
        if importance is None:
            return None
        return {
            f"residual::{feature_name}": value
            for feature_name, value in importance.items()
        }
