from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.base import ComposablePreprocessor


@dataclass
class SNVPreprocessor(ComposablePreprocessor):
    """各スペクトルに Standard Normal Variate を適用する前処理。"""

    def fit(self, ds: Dataset) -> None:
        """学習パラメータを持たないため何もしない。"""
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        """各サンプル行を平均 0、標準偏差 1 に正規化する。"""
        mean = np.mean(ds.X, axis=1, keepdims=True)
        std = np.std(ds.X, axis=1, keepdims=True)
        safe_std = np.where(std == 0.0, 1.0, std)
        transformed = (ds.X - mean) / safe_std
        return ds.with_features(transformed)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """学習不要のためそのまま変換結果を返す。"""
        return self.transform(ds)
