from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pywt  # type: ignore[import-untyped]

from src.data.dataset import Dataset
from src.preprocess.core.base import ComposablePreprocessor


@dataclass
class DWTFeatureExtractor(ComposablePreprocessor):
    """離散ウェーブレット係数をサブバンドごとの要約統計に変換する。"""

    wavelet: str = "db4"
    level: int = 3
    mode: str = "symmetric"
    include_approximation: bool = True
    include_details: bool = True
    summary_stats: list[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max", "energy"]
    )
    coeff_labels_: list[str] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    input_length_: int | None = field(default=None, init=False)
    resolved_level_: int | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """入力次元から使用可能な DWT サブバンド構成を決める。"""
        wavelet = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(ds.X.shape[1], wavelet.dec_len)
        if max_level < 1:
            raise ValueError("DWT を計算できるだけの特徴量長がありません")
        if not self.include_approximation and not self.include_details:
            raise ValueError(
                "include_approximation と include_details のどちらかを有効化してください"
            )
        self._validate_summary_stats()

        self.input_length_ = ds.X.shape[1]
        self.resolved_level_ = min(self.level, max_level)
        self.coeff_labels_ = self._build_coeff_labels()
        self.feature_names_ = self._build_feature_names()

    def transform(self, ds: Dataset) -> Dataset:
        """各サンプルの DWT 要約特徴量を返す。"""
        self._validate_input(ds)
        assert self.resolved_level_ is not None

        transformed = np.stack(
            [self._transform_row(row) for row in ds.X],
            axis=0,
        ).astype(np.float32)
        return ds.with_features(transformed, feature_names=self.feature_names_)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """係数構成を確定して DWT 要約特徴量を返す。"""
        self.fit(ds)
        return self.transform(ds)

    def _transform_row(self, row: np.ndarray) -> np.ndarray:
        """1 サンプル分の DWT 係数を要約統計へ変換する。"""
        coeffs = pywt.wavedec(
            row,
            wavelet=self.wavelet,
            level=self.resolved_level_,
            mode=self.mode,
        )
        summary_values: list[np.float32] = []
        for index, coeff in enumerate(coeffs):
            if index == 0 and not self.include_approximation:
                continue
            if index > 0 and not self.include_details:
                continue
            coeff_array = np.asarray(coeff, dtype=np.float32)
            for stat in self.summary_stats:
                summary_values.append(self._compute_stat(coeff_array, stat))
        return np.asarray(summary_values, dtype=np.float32)

    def _build_coeff_labels(self) -> list[str]:
        """使用する係数配列のラベル一覧を返す。"""
        labels: list[str] = []
        assert self.resolved_level_ is not None

        if self.include_approximation:
            labels.append(f"a{self.resolved_level_}")
        if self.include_details:
            labels.extend(
                f"d{detail_level}"
                for detail_level in range(self.resolved_level_, 0, -1)
            )
        return labels

    def _build_feature_names(self) -> list[str]:
        """DWT 要約特徴量名を生成する。"""
        names: list[str] = []
        for label in self.coeff_labels_:
            names.extend(f"dwt_{label}_{stat}" for stat in self.summary_stats)
        return names

    def _validate_summary_stats(self) -> None:
        """指定された統計量の一覧を検証する。"""
        supported = {"mean", "std", "min", "max", "energy"}
        if not self.summary_stats:
            raise ValueError("summary_stats は 1 つ以上指定してください")
        unknown_stats = [stat for stat in self.summary_stats if stat not in supported]
        if unknown_stats:
            supported_text = ", ".join(sorted(supported))
            unknown_text = ", ".join(unknown_stats)
            raise ValueError(
                f"未対応の summary_stats です: {unknown_text} (allowed: {supported_text})"
            )

    def _compute_stat(self, coeff: np.ndarray, stat: str) -> np.float32:
        """係数列の統計量を 1 つ計算して返す。"""
        if stat == "mean":
            return np.float32(np.mean(coeff))
        if stat == "std":
            return np.float32(np.std(coeff))
        if stat == "min":
            return np.float32(np.min(coeff))
        if stat == "max":
            return np.float32(np.max(coeff))
        if stat == "energy":
            return np.float32(np.mean(np.square(coeff)))
        raise ValueError(f"未対応の統計量です: {stat}")

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時と同じ入力次元だけを受け付ける。"""
        if self.input_length_ is None or self.resolved_level_ is None:
            raise ValueError("先に fit を実行してください")
        if ds.X.shape[1] != self.input_length_:
            raise ValueError("fit 時と異なる特徴量長には変換できません")
