from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

EXCLUDED_TRAIN_SPECIES = frozenset({"ベイスギ", "ベイマツ"})


@dataclass
class Dataset:
    """
    スペクトルデータコンテナ。

    X: 特徴量データ (n_samples, n_features)
    y: 目的変数データ (n_samples,) / test時はNone
    groups: グループデータ (n_samples,)
    sample_id: サンプルIDデータ (n_samples,)
    wavenumbers: 波数データ (n_features,)
    """

    X: np.ndarray
    y: np.ndarray | None
    groups: np.ndarray | None
    sample_id: np.ndarray
    wavenumbers: np.ndarray
    feature_names: np.ndarray | list[str] | None = None

    def __post_init__(self) -> None:
        """配列形状を検証し、特徴量メタデータを補完する。"""
        if self.X.ndim != 2:
            raise ValueError("X は 2 次元配列である必要があります")

        n_samples, n_features = self.X.shape
        if self.sample_id.shape != (n_samples,):
            raise ValueError("sample_id の長さが X と一致しません")
        if self.y is not None and self.y.shape != (n_samples,):
            raise ValueError("y の長さが X と一致しません")
        if self.groups is not None and self.groups.shape != (n_samples,):
            raise ValueError("groups の長さが X と一致しません")

        self.wavenumbers = np.asarray(self.wavenumbers, dtype=np.float32)
        if self.wavenumbers.shape != (n_features,):
            raise ValueError("wavenumbers の長さが特徴量数と一致しません")

        if self.feature_names is None:
            self.feature_names = self._default_feature_names(self.wavenumbers)
            return

        self.feature_names = self._feature_names_array(self.feature_names)
        if self.feature_names.shape != (n_features,):
            raise ValueError("feature_names の長さが特徴量数と一致しません")

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        encoding: str = "cp932",
        target_col: str = "含水率",
        sample_id_col: str = "sample number",
        group_col: str = "species number",
    ) -> Dataset:
        """CSV から `Dataset` を生成する。"""
        with Path(path).open("r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("ヘッダー行が見つかりません")
            fieldnames = reader.fieldnames
            rows = list(reader)

        if sample_id_col not in fieldnames:
            raise ValueError(f"'{sample_id_col}' 列が見つかりません")
        if group_col not in fieldnames:
            raise ValueError(f"'{group_col}' 列が見つかりません")

        ignore_cols = {sample_id_col, group_col, target_col, "樹種"}
        feature_cols = [col for col in fieldnames if col not in ignore_cols]
        if not feature_cols:
            raise ValueError("スペクトル列が見つかりません")

        rows = cls._filter_excluded_train_species(rows, fieldnames, target_col)
        if not rows:
            raise ValueError("有効なデータ行がありません")

        try:
            wavenumbers = np.asarray(feature_cols, dtype=np.float32)
        except ValueError as exc:
            raise ValueError("スペクトル列名を波数に変換できません") from exc

        X = np.asarray(
            [[float(row[col]) for col in feature_cols] for row in rows],
            dtype=np.float32,
        )
        sample_id = np.asarray(
            [int(row[sample_id_col]) for row in rows], dtype=np.int64
        )
        groups = np.asarray([int(row[group_col]) for row in rows], dtype=np.int64)
        y = (
            np.asarray([float(row[target_col]) for row in rows], dtype=np.float32)
            if target_col in fieldnames
            else None
        )

        return cls(
            X=X,
            y=y,
            groups=groups,
            sample_id=sample_id,
            wavenumbers=wavenumbers,
            feature_names=feature_cols,
        )

    @staticmethod
    def _filter_excluded_train_species(
        rows: list[dict[str, str]],
        fieldnames: Sequence[str],
        target_col: str,
    ) -> list[dict[str, str]]:
        """学習データでは対象外樹種を読み込み時に除外する。"""
        if target_col not in fieldnames or "樹種" not in fieldnames:
            return rows
        return [
            row for row in rows if row.get("樹種", "") not in EXCLUDED_TRAIN_SPECIES
        ]

    @classmethod
    def concat(cls, datasets: list[Dataset]) -> Dataset:
        """複数 `Dataset` をサンプル方向に連結する。"""
        if not datasets:
            raise ValueError("datasets が空です")

        ref = datasets[0]

        for ds in datasets:
            if not np.array_equal(ds.wavenumbers, ref.wavenumbers):
                raise ValueError("wavenumbers が一致しません")
            if not np.array_equal(
                cls._feature_names_array(ds.feature_names),
                cls._feature_names_array(ref.feature_names),
            ):
                raise ValueError("feature_names が一致しません")
            if (ds.y is None) != (ref.y is None):
                raise ValueError("y の有無が混在しています")
            if (ds.groups is None) != (ref.groups is None):
                raise ValueError("groups の有無が混在しています")

        return cls(
            X=np.concatenate([ds.X for ds in datasets], axis=0),
            y=(
                None
                if ref.y is None
                else np.concatenate(
                    [ds.y for ds in datasets if ds.y is not None], axis=0
                )
            ),
            groups=(
                None
                if ref.groups is None
                else np.concatenate(
                    [ds.groups for ds in datasets if ds.groups is not None], axis=0
                )
            ),
            sample_id=np.concatenate([ds.sample_id for ds in datasets], axis=0),
            wavenumbers=ref.wavenumbers.copy(),
            feature_names=cls._feature_names_array(ref.feature_names).copy(),
        )

    @classmethod
    def concat_features(cls, datasets: list[Dataset]) -> Dataset:
        """複数 `Dataset` を特徴量方向に連結する。"""
        if not datasets:
            raise ValueError("datasets が空です")

        ref = datasets[0]
        for ds in datasets[1:]:
            if ds.X.shape[0] != ref.X.shape[0]:
                raise ValueError("サンプル数が一致しません")
            if not np.array_equal(ds.sample_id, ref.sample_id):
                raise ValueError("sample_id が一致しません")
            if (ds.y is None) != (ref.y is None):
                raise ValueError("y の有無が混在しています")
            if (
                ds.y is not None
                and ref.y is not None
                and not np.array_equal(ds.y, ref.y)
            ):
                raise ValueError("y が一致しません")
            if (ds.groups is None) != (ref.groups is None):
                raise ValueError("groups の有無が混在しています")
            if (
                ds.groups is not None
                and ref.groups is not None
                and not np.array_equal(ds.groups, ref.groups)
            ):
                raise ValueError("groups が一致しません")

        feature_names = np.concatenate(
            [
                (
                    ds.feature_names
                    if ds.feature_names is not None
                    else cls._default_feature_names(ds.wavenumbers)
                )
                for ds in datasets
            ],
            axis=0,
        )

        return cls(
            X=np.concatenate([ds.X for ds in datasets], axis=1),
            y=None if ref.y is None else ref.y.copy(),
            groups=None if ref.groups is None else ref.groups.copy(),
            sample_id=ref.sample_id.copy(),
            wavenumbers=np.concatenate([ds.wavenumbers for ds in datasets], axis=0),
            feature_names=feature_names,
        )

    def with_features(
        self,
        X: np.ndarray,
        *,
        wavenumbers: np.ndarray | None = None,
        feature_names: np.ndarray | list[str] | None = None,
    ) -> Dataset:
        """同一サンプル集合を保ったまま特徴量だけ差し替える。"""
        n_features = X.shape[1]
        resolved_wavenumbers = (
            self.wavenumbers.copy()
            if wavenumbers is None and n_features == self.X.shape[1]
            else (
                np.full(n_features, np.nan, dtype=np.float32)
                if wavenumbers is None
                else np.asarray(wavenumbers, dtype=np.float32)
            )
        )
        resolved_feature_names = (
            self.feature_names.copy()
            if feature_names is None
            and n_features == self.X.shape[1]
            and self.feature_names is not None
            else feature_names
        )
        return Dataset(
            X=np.asarray(X, dtype=np.float32),
            y=None if self.y is None else self.y.copy(),
            groups=None if self.groups is None else self.groups.copy(),
            sample_id=self.sample_id.copy(),
            wavenumbers=resolved_wavenumbers,
            feature_names=resolved_feature_names,
        )

    def with_target(self, y: np.ndarray | None) -> Dataset:
        """同一サンプル集合を保ったまま目的変数だけ差し替える。"""
        return Dataset(
            X=self.X,
            y=None if y is None else np.asarray(y, dtype=np.float32),
            groups=self.groups,
            sample_id=self.sample_id,
            wavenumbers=self.wavenumbers,
            feature_names=self.feature_names,
        )

    @staticmethod
    def _default_feature_names(wavenumbers: np.ndarray) -> np.ndarray:
        """波数から既定の特徴量名を生成する。"""
        names = [
            str(float(wavenumber)) if np.isfinite(wavenumber) else f"feature_{index}"
            for index, wavenumber in enumerate(wavenumbers)
        ]
        return np.asarray(names, dtype=object)

    @staticmethod
    def _feature_names_array(
        feature_names: np.ndarray | list[str] | None,
    ) -> np.ndarray:
        """特徴量名を object 配列に正規化する。"""
        if feature_names is None:
            raise ValueError("feature_names が未設定です")
        return np.asarray(feature_names, dtype=object)
