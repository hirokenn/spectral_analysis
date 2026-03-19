from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from tqdm.auto import trange  # type: ignore[import-untyped]

from src.data.dataset import Dataset


@dataclass
class LOSOCV:
    """グループ単位の Leave-One-Group-Out（LOSO）分割器。"""

    dataset: Dataset
    show_progress: bool = True

    def __post_init__(self) -> None:
        """入力パラメータを検証する。"""
        if self.dataset.groups is None:
            raise ValueError("LOSOCV には dataset.groups が必要です")
        unique_groups = np.unique(self.dataset.groups)
        if unique_groups.size < 2:
            raise ValueError("LOSO CV には 2 つ以上のグループが必要です")

    def __iter__(self) -> Iterator[tuple[Dataset, Dataset]]:
        """(train_ds, test_ds) を順に返す。"""
        return self.split()

    def split(self) -> Iterator[tuple[Dataset, Dataset]]:
        """各 split で 1 グループだけを test にして分割する。"""
        assert self.dataset.groups is not None
        groups = self.dataset.groups
        unique_groups = np.unique(groups)

        for group in trange(
            unique_groups.size,
            desc="LOSOCV",
            unit="split",
            disable=not self.show_progress,
        ):
            test_group = unique_groups[group]
            test_mask = groups == test_group
            train_mask = ~test_mask
            yield self._subset(train_mask), self._subset(test_mask)

    def _subset(self, mask: np.ndarray) -> Dataset:
        """mask に基づく部分 `Dataset` を返す。"""
        return Dataset(
            X=self.dataset.X[mask],
            y=None if self.dataset.y is None else self.dataset.y[mask],
            groups=None if self.dataset.groups is None else self.dataset.groups[mask],
            sample_id=self.dataset.sample_id[mask],
            wavenumbers=self.dataset.wavenumbers.copy(),
            feature_names=(
                None
                if self.dataset.feature_names is None
                else self.dataset.feature_names.copy()
            ),
        )


@dataclass
class GroupCV:
    """グループ単位で holdout する CV 分割器。"""

    dataset: Dataset
    n_splits: int = 20
    test_size: float = 0.3
    random_state: int | None = 42
    show_progress: bool = True

    def __post_init__(self) -> None:
        """入力パラメータを検証する。"""
        if self.dataset.groups is None:
            raise ValueError("GroupCV には dataset.groups が必要です")
        if self.n_splits < 1:
            raise ValueError("n_splits は 1 以上で指定してください")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size は 0 と 1 の間で指定してください")

        unique_groups = np.unique(self.dataset.groups)
        if unique_groups.size < 2:
            raise ValueError("CV には 2 つ以上のグループが必要です")
        n_test_groups = max(1, int(round(unique_groups.size * self.test_size)))
        if n_test_groups >= unique_groups.size:
            raise ValueError("test_size が大きすぎます")
        if self.n_splits < self._min_splits_for_full_validation(unique_groups.size):
            raise ValueError(
                "全グループを最低1回 validation に含めるには n_splits が不足しています"
            )

    def __iter__(self) -> Iterator[tuple[Dataset, Dataset]]:
        """(train_ds, test_ds) を順に返す。"""
        return self.split()

    def split(self) -> Iterator[tuple[Dataset, Dataset]]:
        """グループ単位で train/test を分割する。"""
        assert self.dataset.groups is not None
        groups = self.dataset.groups
        unique_groups = np.unique(groups)
        n_test_groups = max(1, int(round(unique_groups.size * self.test_size)))
        rng = np.random.default_rng(self.random_state)
        mandatory_test_groups = self._build_mandatory_test_groups(
            unique_groups=unique_groups,
            n_test_groups=n_test_groups,
            rng=rng,
        )
        n_random_splits = self.n_splits - len(mandatory_test_groups)

        for _ in trange(
            len(mandatory_test_groups),
            desc="GroupCV",
            unit="split",
            disable=not self.show_progress,
        ):
            test_groups = mandatory_test_groups[_]
            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask
            yield self._subset(train_mask), self._subset(test_mask)

        for _ in trange(
            n_random_splits,
            desc="GroupCV",
            unit="split",
            disable=not self.show_progress,
        ):
            test_groups = rng.choice(unique_groups, size=n_test_groups, replace=False)
            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask
            yield self._subset(train_mask), self._subset(test_mask)

    def _min_splits_for_full_validation(self, n_groups: int) -> int:
        """全グループを最低1回 validation に含める最小 split 数を返す。"""
        n_test_groups = max(1, int(round(n_groups * self.test_size)))
        return (n_groups + n_test_groups - 1) // n_test_groups

    @staticmethod
    def _build_mandatory_test_groups(
        *,
        unique_groups: np.ndarray,
        n_test_groups: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """全グループを最低1回含む test group 割当を作成する。"""
        shuffled_groups = rng.permutation(unique_groups)
        return [
            shuffled_groups[i : i + n_test_groups]
            for i in range(0, shuffled_groups.size, n_test_groups)
        ]

    def _subset(self, mask: np.ndarray) -> Dataset:
        """mask に基づく部分 `Dataset` を返す。"""
        return Dataset(
            X=self.dataset.X[mask],
            y=None if self.dataset.y is None else self.dataset.y[mask],
            groups=None if self.dataset.groups is None else self.dataset.groups[mask],
            sample_id=self.dataset.sample_id[mask],
            wavenumbers=self.dataset.wavenumbers.copy(),
            feature_names=(
                None
                if self.dataset.feature_names is None
                else self.dataset.feature_names.copy()
            ),
        )
