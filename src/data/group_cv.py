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
            leave=False,
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
    """
    グループ単位の Repeated Group CV 分割器。

    各 repeat でユニークなグループをシャッフルし、順に ``n_splits`` 個の
    fold に分割する。各 fold の test はその fold に割り当てられたグループ
    のサンプルからなる。``n_repeats`` 回繰り返す。

    CVTrainer は同一サンプルが複数 fold に含まれる場合、OOF を平均する。
    """

    dataset: Dataset
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int | None = 42
    show_progress: bool = True

    def __post_init__(self) -> None:
        """入力パラメータを検証する。"""
        if self.dataset.groups is None:
            raise ValueError("GroupCV には dataset.groups が必要です")
        if self.n_splits < 2:
            raise ValueError("n_splits は 2 以上で指定してください")
        if self.n_repeats < 1:
            raise ValueError("n_repeats は 1 以上で指定してください")

        unique_groups = np.unique(self.dataset.groups)
        if unique_groups.size < 2:
            raise ValueError("CV には 2 つ以上のグループが必要です")
        if self.n_splits > unique_groups.size:
            raise ValueError("n_splits はユニークなグループ数以下で指定してください")

    def __iter__(self) -> Iterator[tuple[Dataset, Dataset]]:
        """(train_ds, test_ds) を順に返す。"""
        return self.split()

    def split(self) -> Iterator[tuple[Dataset, Dataset]]:
        """各 repeat で ``n_splits`` 回、グループ単位に train/test を分割する。"""
        assert self.dataset.groups is not None
        groups = self.dataset.groups
        unique_groups = np.unique(groups)
        base_seed = self.random_state if self.random_state is not None else 0

        for repeat_idx in range(self.n_repeats):
            repeat_seed = base_seed + repeat_idx
            rng = np.random.default_rng(repeat_seed)
            shuffled = rng.permutation(unique_groups)
            fold_groups_list = np.array_split(shuffled, self.n_splits)

            desc = f"GroupCV (repeat {repeat_idx + 1}/{self.n_repeats})"
            for fold_idx in trange(
                self.n_splits,
                desc=desc,
                unit="split",
                disable=not self.show_progress,
                leave=False,
            ):
                test_groups = fold_groups_list[fold_idx]
                test_mask = np.isin(groups, test_groups)
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
