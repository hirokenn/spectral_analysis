from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import Dataset
from src.data.group_cv import GroupCV


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
    cv = GroupCV(dataset=ds, n_splits=5, test_size=0.25, random_state=0)

    splits = list(cv)

    assert len(splits) == 5
    for train_ds, test_ds in splits:
        assert train_ds.groups is not None
        assert test_ds.groups is not None
        assert set(train_ds.sample_id).isdisjoint(set(test_ds.sample_id))
        assert set(train_ds.sample_id) | set(test_ds.sample_id) == set(ds.sample_id)
        assert set(train_ds.groups).isdisjoint(set(test_ds.groups))


def test_group_cv_is_reproducible_with_same_seed() -> None:
    ds = make_dataset()
    cv1 = GroupCV(dataset=ds, n_splits=1, test_size=0.5, random_state=7)
    cv2 = GroupCV(dataset=ds, n_splits=1, test_size=0.5, random_state=7)

    (_, test1), (_, test2) = next(iter(cv1)), next(iter(cv2))

    assert np.array_equal(np.sort(test1.sample_id), np.sort(test2.sample_id))


def test_group_cv_raises_when_groups_missing() -> None:
    ds = make_dataset()
    ds.groups = None

    with pytest.raises(ValueError, match="groups"):
        GroupCV(dataset=ds)
