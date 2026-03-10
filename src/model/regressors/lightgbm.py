from __future__ import annotations

from typing import Any

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.data.dataset import Dataset
from src.model.base_model import BaseModel, ModelType


class LightGBMRegressor(BaseModel):
    """`LGBMRegressor` を `Dataset` 入出力で扱うための回帰器。"""

    def __init__(self, **params: Any) -> None:
        """`LGBMRegressor` のパラメータを初期化する。"""
        default_params: dict[str, Any] = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**self.params)
        self._is_fitted = False
        self.feature_names_: list[str] | None = None

    @property
    def model_type(self) -> ModelType:
        """MLflow 連携用のモデル種別を返す。"""
        return ModelType.LIGHTGBM

    def fit(self, ds: Dataset) -> None:
        """学習データでモデルを学習する。"""
        if ds.y is None:
            raise ValueError("学習には y が必要です")
        if ds.feature_names is None:
            raise ValueError("学習には feature_names が必要です")

        self.model = lgb.LGBMRegressor(**self.params)
        feature_names = [str(name) for name in ds.feature_names]
        X = self._to_dataframe(ds.X, feature_names)
        self.model.fit(X, ds.y, feature_name=feature_names)
        self._is_fitted = True
        self.feature_names_ = feature_names

    def predict(self, ds: Dataset) -> np.ndarray:
        """学習済みモデルで予測値を返す。"""
        if not self._is_fitted:
            raise ValueError("モデルが未学習です。先に fit を実行してください")
        feature_names = (
            [str(name) for name in ds.feature_names]
            if ds.feature_names is not None
            else self.feature_names_
        )
        if feature_names is None:
            raise ValueError("予測には feature_names が必要です")
        X = self._to_dataframe(ds.X, feature_names)
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def get_native_model(self) -> Any:
        """内部の `LGBMRegressor` インスタンスを返す。"""
        return self.model

    def set_native_model(self, native_model: Any) -> None:
        """読み込んだ `LGBMRegressor` を設定し、学習済み状態にする。"""
        self.model = native_model
        self._is_fitted = True

    def feature_importance(self) -> dict[str, float] | None:
        """特徴量重要度辞書を返す。"""
        if not self._is_fitted or not hasattr(self.model, "feature_importances_"):
            return None

        importance = np.asarray(self.model.feature_importances_).ravel()
        feature_names = self.feature_names_ or [
            f"feature_{i}" for i in range(importance.shape[0])
        ]
        return {
            feature_name: float(value)
            for feature_name, value in zip(feature_names, importance, strict=True)
        }

    @staticmethod
    def _to_dataframe(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
        """特徴量名付き `DataFrame` に変換する。"""
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names の長さが X の列数と一致しません")
        return pd.DataFrame(X, columns=pd.Index(feature_names))
