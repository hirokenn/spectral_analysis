from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.core.base import ComposablePreprocessor


def _build_interval_indices(
    wavenumbers: np.ndarray,
    interval_size: int,
) -> list[np.ndarray]:
    """波数軸を固定点数ごとの区間インデックスに分割する。"""
    if interval_size < 1:
        raise ValueError("interval_size は 1 以上で指定してください")

    n_features = wavenumbers.shape[0]
    n_intervals = n_features // interval_size
    if n_intervals == 0:
        return [np.arange(n_features, dtype=np.int64)]

    return [
        np.arange(
            interval_id * interval_size,
            (interval_id + 1) * interval_size,
            dtype=np.int64,
        )
        for interval_id in range(n_intervals)
    ]


def _interval_bounds(
    wavenumbers: np.ndarray,
    interval_indices: np.ndarray,
) -> tuple[float, float]:
    """区間に含まれる波数の下限と上限を返す。"""
    values = wavenumbers[interval_indices]
    return float(np.min(values)), float(np.max(values))


@dataclass
class IntervalMeanFeatureExtractor(ComposablePreprocessor):
    """一定点数ごとの平均値を特徴量化する。"""

    interval_size: int = 100
    interval_indices_: list[np.ndarray] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    fitted_wavenumbers_: np.ndarray | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """入力波数軸に基づいて集約区間を定義する。"""
        self.fitted_wavenumbers_ = ds.wavenumbers.copy()
        self.interval_indices_ = _build_interval_indices(
            ds.wavenumbers,
            self.interval_size,
        )
        self.feature_names_ = [
            self._feature_name(ds.wavenumbers, indices)
            for indices in self.interval_indices_
        ]

    def transform(self, ds: Dataset) -> Dataset:
        """各区間の平均値を並べた特徴量を返す。"""
        self._validate_input(ds)
        features = np.stack(
            [np.mean(ds.X[:, indices], axis=1) for indices in self.interval_indices_],
            axis=1,
        ).astype(np.float32)
        return ds.with_features(features, feature_names=self.feature_names_)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """区間定義を作成して平均特徴量を返す。"""
        self.fit(ds)
        return self.transform(ds)

    @staticmethod
    def _feature_name(wavenumbers: np.ndarray, interval_indices: np.ndarray) -> str:
        """区間平均の特徴量名を生成する。"""
        lower, upper = _interval_bounds(wavenumbers, interval_indices)
        return f"interval_mean_{upper:.1f}_{lower:.1f}"

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時の波数軸と一致する入力だけを受け付ける。"""
        if self.fitted_wavenumbers_ is None or not self.interval_indices_:
            raise ValueError("先に fit を実行してください")
        if not np.array_equal(ds.wavenumbers, self.fitted_wavenumbers_):
            raise ValueError("fit 時と異なる wavenumbers には変換できません")


@dataclass
class IntervalSlopeFeatureExtractor(ComposablePreprocessor):
    """一定点数ごとの線形傾きを特徴量化する。"""

    interval_size: int = 100
    interval_indices_: list[np.ndarray] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    centered_wavenumbers_: list[np.ndarray] = field(default_factory=list, init=False)
    variance_: list[float] = field(default_factory=list, init=False)
    fitted_wavenumbers_: np.ndarray | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """入力波数軸に基づいて傾き計算用の係数を準備する。"""
        self.fitted_wavenumbers_ = ds.wavenumbers.copy()
        self.interval_indices_ = _build_interval_indices(
            ds.wavenumbers,
            self.interval_size,
        )
        self.feature_names_ = [
            self._feature_name(ds.wavenumbers, indices)
            for indices in self.interval_indices_
        ]
        self.centered_wavenumbers_ = []
        self.variance_ = []
        for indices in self.interval_indices_:
            x = ds.wavenumbers[indices].astype(np.float32)
            centered = x - np.mean(x)
            variance = float(np.sum(centered**2))
            self.centered_wavenumbers_.append(centered)
            self.variance_.append(variance if variance > 0.0 else 1.0)

    def transform(self, ds: Dataset) -> Dataset:
        """各区間内の一次回帰傾きを特徴量として返す。"""
        self._validate_input(ds)
        slopes = []
        for indices, centered_x, variance in zip(
            self.interval_indices_,
            self.centered_wavenumbers_,
            self.variance_,
            strict=True,
        ):
            y = ds.X[:, indices]
            centered_y = y - np.mean(y, axis=1, keepdims=True)
            slope = np.sum(centered_y * centered_x[None, :], axis=1) / variance
            slopes.append(slope)
        features = np.stack(slopes, axis=1).astype(np.float32)
        return ds.with_features(features, feature_names=self.feature_names_)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """傾き計算用の係数を準備して特徴量を返す。"""
        self.fit(ds)
        return self.transform(ds)

    @staticmethod
    def _feature_name(wavenumbers: np.ndarray, interval_indices: np.ndarray) -> str:
        """区間傾きの特徴量名を生成する。"""
        lower, upper = _interval_bounds(wavenumbers, interval_indices)
        return f"interval_slope_{upper:.1f}_{lower:.1f}"

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時の波数軸と一致する入力だけを受け付ける。"""
        if self.fitted_wavenumbers_ is None or not self.interval_indices_:
            raise ValueError("先に fit を実行してください")
        if not np.array_equal(ds.wavenumbers, self.fitted_wavenumbers_):
            raise ValueError("fit 時と異なる wavenumbers には変換できません")
