from __future__ import annotations

from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.pipeline import PreprocessingPipeline
from src.recipes.base import Recipe


def make_identity_pipeline() -> PreprocessingPipeline:
    """既定の no-op 前処理パイプラインを返す。"""
    return PreprocessingPipeline() >> IdentityPreprocessor()


RECIPES: dict[str, Recipe] = {
    "base_pls": Recipe(
        name="base_pls",
        preprocessor_factory=make_identity_pipeline,
        model_name="base_pls",
    ),
}
