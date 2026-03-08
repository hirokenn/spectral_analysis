from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]

from src.data.dataset import Dataset
from src.model.base_model import BaseModel, ModelType


class PLSRegressor(BaseModel):
    """`PLSRegression` を `Dataset` 入出力で扱うための回帰器。"""

    def __init__(self, **params: Any) -> None:
        """`PLSRegression` のパラメータを可変長引数で受け取る。"""
        default_params: dict[str, Any] = {"n_components": 8, "scale": True}
        default_params.update(params)

        if int(default_params["n_components"]) < 1:
            raise ValueError("n_components は 1 以上で指定してください")

        self.params = default_params
        self.model = PLSRegression(**self.params)
        self._is_fitted = False

    @property
    def model_type(self) -> ModelType:
        return ModelType.SKLEARN

    def fit(self, ds: Dataset) -> None:
        """学習データでモデルを学習する。"""
        if ds.y is None:
            raise ValueError("学習には y が必要です")
        if ds.X.ndim != 2:
            raise ValueError("X は 2 次元配列である必要があります")

        requested = int(self.params["n_components"])
        effective_components = min(requested, ds.X.shape[0] - 1, ds.X.shape[1])
        if effective_components < 1:
            raise ValueError("学習に必要なサンプル数または特徴量数が不足しています")

        fit_params = {**self.params, "n_components": effective_components}
        self.model = PLSRegression(**fit_params)
        self.model.fit(ds.X, ds.y)
        self._is_fitted = True

    def predict(self, ds: Dataset) -> np.ndarray:
        """学習済みモデルで予測値を返す。"""
        if not self._is_fitted:
            raise ValueError("モデルが未学習です。先に fit を実行してください")
        return self.model.predict(ds.X).ravel()

    def get_native_model(self) -> Any:
        """内部の `PLSRegression` インスタンスを返す。"""
        return self.model

    def set_native_model(self, native_model: Any) -> None:
        """読み込んだ `PLSRegression` を設定し、学習済み状態にする。"""
        self.model = native_model
        self._is_fitted = True

    def feature_importance(self) -> dict[str, float] | None:
        """回帰係数から特徴量重要度辞書を返す。"""
        if not self._is_fitted or not hasattr(self.model, "coef_"):
            return None

        coef = np.asarray(self.model.coef_).ravel()
        return {f"feature_{i}": float(value) for i, value in enumerate(coef)}
