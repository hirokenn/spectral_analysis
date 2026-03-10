from __future__ import annotations

import pytest

from src.preprocess.core.feature_union import FeatureUnion
from src.preprocess.core.pipeline import PreprocessingPipeline
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder
from src.preprocess.features.dwt_features import DWTFeatureExtractor
from src.preprocess.features.group_sequence_features import (
    GroupSequenceFeatureExtractor,
)
from src.preprocess.features.interval_features import (
    IntervalMeanFeatureExtractor,
    IntervalSlopeFeatureExtractor,
)
from src.preprocess.features.water_band_summary import WaterBandSummaryFeatureExtractor
from src.preprocess.transforms.identity import IdentityPreprocessor
from src.preprocess.transforms.savgol import SavitzkyGolayPreprocessor
from src.preprocess.transforms.snv import SNVPreprocessor


def test_build_returns_identity_pipeline_from_config() -> None:
    builder = PreprocessorBuilder(
        config={
            "identity": {"build_type": "identity", "params": {}},
            "base_pipeline": {"build_type": "pipeline", "steps": ["identity"]},
        }
    )

    preprocessor = builder.build("base_pipeline")

    assert isinstance(preprocessor, PreprocessingPipeline)
    assert len(preprocessor.steps) == 1
    assert isinstance(preprocessor.steps[0], IdentityPreprocessor)


def test_build_returns_feature_union_from_config() -> None:
    builder = PreprocessorBuilder(
        config={
            "identity": {"build_type": "identity", "params": {}},
            "union": {
                "build_type": "feature_union",
                "branches": ["identity", "identity"],
            },
        }
    )

    preprocessor = builder.build("union")

    assert isinstance(preprocessor, FeatureUnion)
    assert len(preprocessor.branches) == 2
    assert all(
        isinstance(branch, IdentityPreprocessor) for branch in preprocessor.branches
    )


def test_build_raises_when_preprocessor_name_is_missing() -> None:
    builder = PreprocessorBuilder(
        config={"identity": {"build_type": "identity", "params": {}}}
    )

    with pytest.raises(ValueError, match="前処理定義が見つかりません"):
        builder.build("unknown_preprocessor")


def test_build_raises_when_preprocessor_definitions_cycle() -> None:
    builder = PreprocessorBuilder(
        config={
            "a": {"build_type": "pipeline", "steps": ["b"]},
            "b": {"build_type": "pipeline", "steps": ["a"]},
        }
    )

    with pytest.raises(ValueError, match="循環参照"):
        builder.build("a")


def test_build_returns_extended_preprocessors_from_config() -> None:
    builder = PreprocessorBuilder(
        config={
            "snv": {"build_type": "snv", "params": {}},
            "savgol": {
                "build_type": "savgol",
                "params": {"window_length": 5, "polyorder": 2},
            },
            "group_sequence": {
                "build_type": "group_sequence",
                "params": {"target_wavenumbers": [5200.0, 7000.0], "rolling_window": 3},
            },
            "interval_mean": {
                "build_type": "interval_mean",
                "params": {"interval_size": 2},
            },
            "interval_slope": {
                "build_type": "interval_slope",
                "params": {"interval_size": 2},
            },
            "dwt": {"build_type": "dwt", "params": {"wavelet": "db1", "level": 1}},
            "water": {
                "build_type": "water_band_summary",
                "params": {
                    "bands": [{"name": "band", "lower": 5000.0, "upper": 5400.0}],
                    "stats": ["mean"],
                },
            },
        }
    )

    assert isinstance(builder.build("snv"), SNVPreprocessor)
    assert isinstance(builder.build("savgol"), SavitzkyGolayPreprocessor)
    assert isinstance(builder.build("group_sequence"), GroupSequenceFeatureExtractor)
    assert isinstance(builder.build("interval_mean"), IntervalMeanFeatureExtractor)
    assert isinstance(builder.build("interval_slope"), IntervalSlopeFeatureExtractor)
    assert isinstance(builder.build("dwt"), DWTFeatureExtractor)
    assert isinstance(builder.build("water"), WaterBandSummaryFeatureExtractor)


def test_build_resolves_extends_and_merges_preprocessor_params() -> None:
    builder = PreprocessorBuilder(
        config={
            "_shared_interval": {"params": {"interval_size": 4}},
            "interval_mean": {
                "extends": "_shared_interval",
                "build_type": "interval_mean",
            },
        }
    )

    preprocessor = builder.build("interval_mean")

    assert isinstance(preprocessor, IntervalMeanFeatureExtractor)
    assert preprocessor.interval_size == 4
