from __future__ import annotations

import pytest

from src.preprocess.feature_union import FeatureUnion
from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.pipeline import PreprocessingPipeline
from src.preprocess.preprocessor_builder import PreprocessorBuilder


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
