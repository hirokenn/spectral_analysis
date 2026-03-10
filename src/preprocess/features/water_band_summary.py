from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.core.base import ComposablePreprocessor

DEFAULT_WATER_BANDS: tuple[dict[str, Any], ...] = (
    {"name": "water_7000", "lower": 6800.0, "upper": 7200.0},
    {"name": "water_5200", "lower": 5000.0, "upper": 5400.0},
)
DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "area")


@dataclass(frozen=True)
class WaterBand:
    """水関連帯域の定義。"""

    name: str
    lower: float
    upper: float


@dataclass
class WaterBandSummaryFeatureExtractor(ComposablePreprocessor):
    """水関連帯域ごとの要約統計量を特徴量化する。"""

    bands: list[dict[str, Any]] | None = None
    stats: list[str] | None = None
    resolved_bands_: list[WaterBand] = field(default_factory=list, init=False)
    band_indices_: list[np.ndarray] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    fitted_wavenumbers_: np.ndarray | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """指定帯域の列位置と特徴量名を確定する。"""
        self.fitted_wavenumbers_ = ds.wavenumbers.copy()
        self.resolved_bands_ = self._resolve_bands()
        self.band_indices_ = []
        self.feature_names_ = []

        for band in self.resolved_bands_:
            indices = np.flatnonzero(
                (ds.wavenumbers >= band.lower) & (ds.wavenumbers <= band.upper)
            )
            if indices.size == 0:
                raise ValueError(f"水関連帯域に該当する波数がありません: {band.name}")
            self.band_indices_.append(indices)
            self.feature_names_.extend(
                f"{band.name}_{stat}" for stat in self._resolve_stats()
            )

    def transform(self, ds: Dataset) -> Dataset:
        """各帯域の要約統計量を横連結した特徴量を返す。"""
        self._validate_input(ds)
        stats = self._resolve_stats()
        features: list[np.ndarray] = []

        for band, indices in zip(self.resolved_bands_, self.band_indices_, strict=True):
            _ = band
            segment = ds.X[:, indices]
            band_wavenumbers = ds.wavenumbers[indices]
            for stat in stats:
                features.append(self._compute_stat(segment, band_wavenumbers, stat))

        transformed = np.stack(features, axis=1).astype(np.float32)
        return ds.with_features(transformed, feature_names=self.feature_names_)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """帯域定義を確定して要約統計量特徴量を返す。"""
        self.fit(ds)
        return self.transform(ds)

    def _resolve_bands(self) -> list[WaterBand]:
        """外部入力を `WaterBand` に正規化する。"""
        raw_bands = (
            list(self.bands) if self.bands is not None else list(DEFAULT_WATER_BANDS)
        )
        resolved: list[WaterBand] = []
        for raw_band in raw_bands:
            name = str(raw_band["name"])
            lower = float(raw_band["lower"])
            upper = float(raw_band["upper"])
            if lower >= upper:
                raise ValueError(
                    f"帯域の lower は upper 未満である必要があります: {name}"
                )
            resolved.append(WaterBand(name=name, lower=lower, upper=upper))
        return resolved

    def _resolve_stats(self) -> list[str]:
        """使用する要約統計量名を返す。"""
        stats = list(self.stats) if self.stats is not None else list(DEFAULT_STATS)
        allowed = {"mean", "std", "min", "max", "area"}
        invalid = [stat for stat in stats if stat not in allowed]
        if invalid:
            joined = ", ".join(invalid)
            raise ValueError(f"未対応の統計量です: {joined}")
        return stats

    @staticmethod
    def _compute_stat(
        segment: np.ndarray,
        wavenumbers: np.ndarray,
        stat: str,
    ) -> np.ndarray:
        """指定された統計量を 1 本のベクトルとして計算する。"""
        if stat == "mean":
            return np.mean(segment, axis=1)
        if stat == "std":
            return np.std(segment, axis=1)
        if stat == "min":
            return np.min(segment, axis=1)
        if stat == "max":
            return np.max(segment, axis=1)
        if stat == "area":
            return np.asarray(
                [np.trapezoid(sample[::-1], x=wavenumbers[::-1]) for sample in segment],
                dtype=np.float32,
            )
        raise ValueError(f"未対応の統計量です: {stat}")

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時の波数軸と一致する入力だけを受け付ける。"""
        if self.fitted_wavenumbers_ is None or not self.band_indices_:
            raise ValueError("先に fit を実行してください")
        if not np.array_equal(ds.wavenumbers, self.fitted_wavenumbers_):
            raise ValueError("fit 時と異なる wavenumbers には変換できません")
