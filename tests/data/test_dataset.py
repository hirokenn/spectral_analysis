from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import Dataset


def test_from_csv_reads_train_like_data(tmp_path) -> None:
    csv_path = tmp_path / "train_like.csv"
    csv_path.write_text(
        "sample number,species number,樹種,含水率,10000.0,9990.0\n"
        "1,10,スギ,12.5,0.1,0.2\n"
        "2,11,ヒノキ,20.0,0.3,0.4\n",
        encoding="cp932",
    )

    ds = Dataset.from_csv(csv_path)

    assert ds.X.shape == (2, 2)
    assert ds.y is not None
    assert ds.y.shape == (2,)
    assert ds.groups is not None
    assert np.array_equal(ds.sample_id, np.array([1, 2]))
    assert np.allclose(ds.wavenumbers, np.array([10000.0, 9990.0], dtype=np.float32))


def test_from_csv_reads_test_like_data_without_target(tmp_path) -> None:
    csv_path = tmp_path / "test_like.csv"
    csv_path.write_text(
        "sample number,species number,樹種,10000.0,9990.0\n"
        "101,2,クスノキ,0.11,0.22\n",
        encoding="cp932",
    )

    ds = Dataset.from_csv(csv_path)

    assert ds.y is None
    assert ds.X.shape == (1, 2)
    assert ds.groups is not None
    assert np.array_equal(ds.sample_id, np.array([101]))


def test_from_csv_excludes_beysugi_and_beimatsu_from_train_like_data(tmp_path) -> None:
    csv_path = tmp_path / "train_filtered.csv"
    csv_path.write_text(
        "sample number,species number,樹種,含水率,10000.0,9990.0\n"
        "1,10,スギ,12.5,0.1,0.2\n"
        "2,15,ベイスギ,20.0,0.3,0.4\n"
        "3,17,ベイマツ,30.0,0.5,0.6\n"
        "4,14,ヒノキ,40.0,0.7,0.8\n",
        encoding="cp932",
    )

    ds = Dataset.from_csv(csv_path)

    assert ds.X.shape == (2, 2)
    assert ds.y is not None
    assert np.array_equal(ds.sample_id, np.array([1, 4]))
    assert ds.groups is not None
    assert np.array_equal(ds.groups, np.array([10, 14]))


def test_concat_merges_rows() -> None:
    ds1 = Dataset(
        X=np.array([[0.1, 0.2]], dtype=np.float32),
        y=np.array([1.0], dtype=np.float32),
        groups=np.array([1]),
        sample_id=np.array([10]),
        wavenumbers=np.array([10000.0, 9990.0], dtype=np.float32),
    )
    ds2 = Dataset(
        X=np.array([[0.3, 0.4]], dtype=np.float32),
        y=np.array([2.0], dtype=np.float32),
        groups=np.array([2]),
        sample_id=np.array([11]),
        wavenumbers=np.array([10000.0, 9990.0], dtype=np.float32),
    )

    merged = Dataset.concat([ds1, ds2])

    assert merged.X.shape == (2, 2)
    assert merged.y is not None
    assert np.array_equal(merged.y, np.array([1.0, 2.0], dtype=np.float32))
    assert merged.groups is not None
    assert np.array_equal(merged.groups, np.array([1, 2]))
    assert np.array_equal(merged.sample_id, np.array([10, 11]))


def test_concat_raises_on_mismatched_wavenumbers() -> None:
    ds1 = Dataset(
        X=np.array([[0.1]], dtype=np.float32),
        y=np.array([1.0], dtype=np.float32),
        groups=np.array([1]),
        sample_id=np.array([1]),
        wavenumbers=np.array([10000.0], dtype=np.float32),
    )
    ds2 = Dataset(
        X=np.array([[0.2]], dtype=np.float32),
        y=np.array([2.0], dtype=np.float32),
        groups=np.array([2]),
        sample_id=np.array([2]),
        wavenumbers=np.array([9990.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="wavenumbers"):
        Dataset.concat([ds1, ds2])
