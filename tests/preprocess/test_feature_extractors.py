from __future__ import annotations

import numpy as np

from src.data.dataset import Dataset
from src.preprocess.dwt_features import DWTFeatureExtractor
from src.preprocess.interval_features import (
    IntervalMeanFeatureExtractor,
    IntervalSlopeFeatureExtractor,
)
from src.preprocess.savgol import SavitzkyGolayPreprocessor
from src.preprocess.snv import SNVPreprocessor
from src.preprocess.water_band_summary import WaterBandSummaryFeatureExtractor
from tests.helpers import make_dataset


def make_spectrum_dataset() -> Dataset:
    """スペクトル前処理テスト用の `Dataset` を返す。"""
    wavenumbers = np.array(
        [7200.0, 7100.0, 7000.0, 6900.0, 5300.0, 5200.0, 5100.0, 5000.0],
        dtype=np.float32,
    )
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 1.0, 1.5, 2.0, 2.5],
        ],
        dtype=np.float32,
    )
    return make_dataset(
        X=X,
        y=np.array([1.0, 2.0], dtype=np.float32),
        sample_id=np.array([1, 2], dtype=np.int64),
        groups=np.array([1, 2], dtype=np.int64),
        feature_names=[str(value) for value in wavenumbers],
    ).with_features(
        X, wavenumbers=wavenumbers, feature_names=[str(value) for value in wavenumbers]
    )


def test_snv_preprocessor_normalizes_each_row() -> None:
    ds = make_spectrum_dataset()
    preprocessor = SNVPreprocessor()

    transformed = preprocessor.fit_transform(ds)

    assert np.allclose(np.mean(transformed.X, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.std(transformed.X, axis=1), 1.0, atol=1e-6)


def test_savgol_preprocessor_smooths_each_row() -> None:
    ds = make_spectrum_dataset()
    preprocessor = SavitzkyGolayPreprocessor(window_length=5, polyorder=2)

    transformed = preprocessor.fit_transform(ds)

    assert transformed.X.shape == ds.X.shape
    assert transformed.feature_names is not None
    assert np.array_equal(transformed.feature_names, ds.feature_names)
    assert not np.allclose(transformed.X, ds.X)


def test_interval_feature_extractors_return_expected_feature_count() -> None:
    ds = make_spectrum_dataset()
    mean_extractor = IntervalMeanFeatureExtractor(interval_size=2)
    slope_extractor = IntervalSlopeFeatureExtractor(interval_size=2)

    mean_features = mean_extractor.fit_transform(ds)
    slope_features = slope_extractor.fit_transform(ds)

    assert mean_features.X.shape == (2, 4)
    assert slope_features.X.shape == (2, 4)
    assert all(
        name.startswith("interval_mean_") for name in mean_features.feature_names
    )
    assert all(
        name.startswith("interval_slope_") for name in slope_features.feature_names
    )


def test_dwt_feature_extractor_returns_summary_statistics() -> None:
    ds = make_spectrum_dataset()
    extractor = DWTFeatureExtractor(
        wavelet="db1",
        level=2,
        summary_stats=["mean", "std", "energy"],
    )

    transformed = extractor.fit_transform(ds)

    assert transformed.X.shape[0] == ds.X.shape[0]
    assert transformed.X.shape == (2, 9)
    assert transformed.X.shape[1] == len(transformed.feature_names)
    assert list(transformed.feature_names[:3]) == [
        "dwt_a2_mean",
        "dwt_a2_std",
        "dwt_a2_energy",
    ]


def test_water_band_summary_feature_extractor_returns_selected_stats() -> None:
    ds = make_spectrum_dataset()
    extractor = WaterBandSummaryFeatureExtractor(
        bands=[
            {"name": "water_7000", "lower": 6800.0, "upper": 7200.0},
            {"name": "water_5200", "lower": 5000.0, "upper": 5300.0},
        ],
        stats=["mean", "max", "area"],
    )

    transformed = extractor.fit_transform(ds)

    assert transformed.X.shape == (2, 6)
    assert list(transformed.feature_names) == [
        "water_7000_mean",
        "water_7000_max",
        "water_7000_area",
        "water_5200_mean",
        "water_5200_max",
        "water_5200_area",
    ]
