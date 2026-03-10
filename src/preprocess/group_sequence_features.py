from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.base import ComposablePreprocessor


@dataclass
class GroupSequenceFeatureExtractor(ComposablePreprocessor):
    """グループ内の `sample_id` 順序から時系列特徴量を作る前処理。"""

    target_wavenumbers: tuple[float, ...] = (5200.0, 7000.0)
    rolling_window: int = 5
    feature_names_: list[str] = field(default_factory=list, init=False)
    target_indices_: list[int] = field(default_factory=list, init=False)
    fitted_wavenumbers_: np.ndarray | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """対象波数の列位置と特徴量名を確定する。"""
        if ds.groups is None:
            raise ValueError("グループ特徴量には dataset.groups が必要です")
        if self.rolling_window < 1:
            raise ValueError("rolling_window は 1 以上で指定してください")

        self.fitted_wavenumbers_ = ds.wavenumbers.copy()
        self.target_indices_ = [
            int(np.argmin(np.abs(ds.wavenumbers - target)))
            for target in self.target_wavenumbers
        ]

        feature_names = ["group_position_index", "group_position_ratio"]
        feature_names.extend(
            f"delta_prev_{int(target)}" for target in self.target_wavenumbers
        )
        feature_names.append("delta_prev_global_mean")
        for target in self.target_wavenumbers:
            feature_names.append(f"rolling_mean_{self.rolling_window}_{int(target)}")
            feature_names.append(f"rolling_std_{self.rolling_window}_{int(target)}")
        feature_names.append(f"rolling_mean_{self.rolling_window}_global_mean")
        feature_names.append(f"rolling_std_{self.rolling_window}_global_mean")
        self.feature_names_ = feature_names

    def transform(self, ds: Dataset) -> Dataset:
        """グループ内順序に基づく特徴量を返す。"""
        self._validate_input(ds)
        assert ds.groups is not None

        n_samples = ds.X.shape[0]
        n_targets = len(self.target_indices_)

        position_index = np.zeros(n_samples, dtype=np.float32)
        position_ratio = np.zeros(n_samples, dtype=np.float32)
        delta_prev = np.zeros((n_samples, n_targets + 1), dtype=np.float32)
        rolling_mean = np.zeros((n_samples, n_targets + 1), dtype=np.float32)
        rolling_std = np.zeros((n_samples, n_targets + 1), dtype=np.float32)

        series = np.column_stack(
            [
                ds.X[:, self.target_indices_],
                np.mean(ds.X, axis=1, keepdims=False),
            ]
        ).astype(np.float32)

        for group in np.unique(ds.groups):
            group_indices = np.flatnonzero(ds.groups == group)
            ordered = group_indices[
                np.argsort(ds.sample_id[group_indices], kind="stable")
            ]
            group_size = ordered.size
            if group_size == 0:
                continue

            for position, sample_index in enumerate(ordered):
                position_index[sample_index] = float(position + 1)
                position_ratio[sample_index] = (
                    0.0 if group_size == 1 else float(position / (group_size - 1))
                )

                current = series[sample_index]
                if position > 0:
                    previous_index = ordered[position - 1]
                    delta_prev[sample_index] = current - series[previous_index]

                window_start = max(0, position - self.rolling_window + 1)
                window_indices = ordered[window_start : position + 1]
                window_values = series[window_indices]
                rolling_mean[sample_index] = np.mean(window_values, axis=0)
                rolling_std[sample_index] = np.std(window_values, axis=0)

        features = np.column_stack(
            [
                position_index,
                position_ratio,
                delta_prev,
                rolling_mean,
                rolling_std,
            ]
        ).astype(np.float32)
        return ds.with_features(features, feature_names=self.feature_names_)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """対象波数を確定して特徴量を返す。"""
        self.fit(ds)
        return self.transform(ds)

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時の波数軸と一致する入力だけを受け付ける。"""
        if ds.groups is None:
            raise ValueError("グループ特徴量には dataset.groups が必要です")
        if self.fitted_wavenumbers_ is None or not self.feature_names_:
            raise ValueError("先に fit を実行してください")
        if not np.array_equal(ds.wavenumbers, self.fitted_wavenumbers_):
            raise ValueError("fit 時と異なる wavenumbers には変換できません")
