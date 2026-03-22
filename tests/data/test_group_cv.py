from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import Dataset
from src.data.group_cv import GroupCV, LOSOCV


def make_dataset() -> Dataset:
    """GroupCV テスト用の不均一グループデータを作る。"""
    return Dataset(
        X=np.arange(30, dtype=np.float32).reshape(10, 3),
        y=np.arange(10, dtype=np.float32),
        groups=np.array([1, 1, 1, 2, 2, 3, 4, 4, 4, 4], dtype=np.int64),
        sample_id=np.arange(100, 110, dtype=np.int64),
        wavenumbers=np.array([10000.0, 9990.0, 9980.0], dtype=np.float32),
    )


def test_group_cv_yields_train_test_pairs() -> None:
    ds = make_dataset()
    cv = GroupCV(
        dataset=ds,
        n_splits=4,
        n_repeats=1,
        random_state=0,
        show_progress=False,
    )

    splits = list(cv)

    assert len(splits) == 4
    for train_ds, test_ds in splits:
        assert train_ds.groups is not None
        assert test_ds.groups is not None
        assert set(train_ds.sample_id).isdisjoint(set(test_ds.sample_id))
        assert set(train_ds.sample_id) | set(test_ds.sample_id) == set(ds.sample_id)
        assert set(train_ds.groups).isdisjoint(set(test_ds.groups))


def test_group_cv_is_reproducible_with_same_seed() -> None:
    ds = make_dataset()
    cv1 = GroupCV(
        dataset=ds,
        n_splits=2,
        n_repeats=1,
        random_state=7,
        show_progress=False,
    )
    cv2 = GroupCV(
        dataset=ds,
        n_splits=2,
        n_repeats=1,
        random_state=7,
        show_progress=False,
    )

    (_, test1), (_, test2) = next(iter(cv1)), next(iter(cv2))

    assert np.array_equal(np.sort(test1.sample_id), np.sort(test2.sample_id))


def test_group_cv_covers_all_groups_at_least_once() -> None:
    ds = make_dataset()
    cv = GroupCV(
        dataset=ds,
        n_splits=2,
        n_repeats=1,
        random_state=7,
        show_progress=False,
    )

    tested_groups: set[int] = set()
    for _, test_ds in cv:
        assert test_ds.groups is not None
        tested_groups.update(int(group) for group in np.unique(test_ds.groups))

    assert tested_groups == {1, 2, 3, 4}


def test_group_cv_raises_when_n_splits_too_small() -> None:
    ds = make_dataset()
    with pytest.raises(ValueError, match="n_splits は 2 以上"):
        GroupCV(
            dataset=ds,
            n_splits=1,
            n_repeats=1,
            show_progress=False,
        )


def test_group_cv_raises_when_n_splits_exceeds_group_count() -> None:
    ds = make_dataset()
    with pytest.raises(ValueError, match="ユニークなグループ数以下"):
        GroupCV(
            dataset=ds,
            n_splits=5,
            n_repeats=1,
            show_progress=False,
        )


def test_group_cv_raises_when_groups_missing() -> None:
    ds = make_dataset()
    ds.groups = None

    with pytest.raises(ValueError, match="groups"):
        GroupCV(dataset=ds)


def test_group_cv_repeated_yields_n_repeats_times_splits() -> None:
    """各 repeat で n_splits 回、合計 n_repeats * n_splits 回になる。"""
    ds = make_dataset()
    cv = GroupCV(
        dataset=ds,
        n_splits=4,
        n_repeats=3,
        random_state=42,
        show_progress=False,
    )

    splits = list(cv)

    assert len(splits) == 4 * 3
    for train_ds, test_ds in splits:
        assert set(train_ds.sample_id).isdisjoint(set(test_ds.sample_id))
        assert set(train_ds.sample_id) | set(test_ds.sample_id) == set(ds.sample_id)


def test_group_cv_raises_when_n_repeats_invalid() -> None:
    ds = make_dataset()
    with pytest.raises(ValueError, match="n_repeats"):
        GroupCV(
            dataset=ds,
            n_splits=4,
            n_repeats=0,
            show_progress=False,
        )


def test_loso_cv_yields_one_group_per_split() -> None:
    ds = make_dataset()
    cv = LOSOCV(dataset=ds, show_progress=False)

    splits = list(cv)

    assert len(splits) == 4
    for train_ds, test_ds in splits:
        assert train_ds.groups is not None
        assert test_ds.groups is not None
        assert np.unique(test_ds.groups).size == 1
        assert set(train_ds.groups).isdisjoint(set(test_ds.groups))
        assert set(train_ds.sample_id).isdisjoint(set(test_ds.sample_id))


def test_loso_cv_covers_all_groups_exactly_once() -> None:
    ds = make_dataset()
    cv = LOSOCV(dataset=ds, show_progress=False)

    tested_groups: list[int] = []
    for _, test_ds in cv:
        assert test_ds.groups is not None
        tested_groups.append(int(np.unique(test_ds.groups)[0]))

    assert set(tested_groups) == {1, 2, 3, 4}
    assert len(tested_groups) == len(set(tested_groups))
