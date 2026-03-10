from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from src.data.dataset import Dataset
from src.preprocess.core.base import ComposablePreprocessor


@dataclass
class SavitzkyGolayPreprocessor(ComposablePreprocessor):
    """スペクトルに Savitzky-Golay 平滑化を適用する前処理。"""

    window_length: int = 11
    polyorder: int = 3
    mode: str = "interp"

    def fit(self, ds: Dataset) -> None:
        """学習パラメータを持たないため何もしない。"""
        _ = ds
        self._validate_params()

    def transform(self, ds: Dataset) -> Dataset:
        """各サンプル行に Savitzky-Golay 平滑化を適用する。"""
        self._validate_params()
        if self.window_length > ds.X.shape[1]:
            raise ValueError("window_length は特徴量数以下で指定してください")
        smoothed = savgol_filter(
            ds.X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            axis=1,
            mode=self.mode,
        ).astype(np.float32)
        return ds.with_features(smoothed)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """学習不要のためそのまま変換結果を返す。"""
        return self.transform(ds)

    def _validate_params(self) -> None:
        """パラメータの妥当性を検証する。"""
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError("window_length は 3 以上の奇数で指定してください")
        if self.polyorder < 0:
            raise ValueError("polyorder は 0 以上で指定してください")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder は window_length より小さくしてください")
