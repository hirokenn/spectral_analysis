from __future__ import annotations

import pytest

from src.recipes.builder import RecipeBuilder
from tests.helpers import make_recipe


def test_recipe_builder_returns_requested_recipe() -> None:
    builder = RecipeBuilder(recipes={"dummy_recipe": make_recipe()})

    recipe = builder.build("dummy_recipe")

    assert recipe.name == "dummy_recipe"
    assert recipe.model_name == "dummy_model"


def test_recipe_builder_raises_for_unknown_recipe() -> None:
    builder = RecipeBuilder(recipes={"dummy_recipe": make_recipe()})

    with pytest.raises(ValueError, match="レシピ定義が見つかりません"):
        builder.build("unknown_recipe")


def test_recipe_builder_contains_snv_pls_in_default_recipes() -> None:
    """デフォルトレシピに snv_pls が含まれていることを確認する。"""
    recipe = RecipeBuilder().build("snv_pls")

    assert recipe.name == "snv_pls"
    assert recipe.preprocessor_name == "snv_pls_pipeline"
    assert recipe.model_name == "snv_pls"


def test_recipe_builder_contains_snv_lgbm_log1p_in_default_recipes() -> None:
    """デフォルトレシピに log1p 付きレシピが含まれていることを確認する。"""
    recipe = RecipeBuilder().build("snv_lgbm_log1p")

    assert recipe.name == "snv_lgbm_log1p"
    assert recipe.preprocessor_name == "snv_lgbm_pipeline"
    assert recipe.model_name == "snv_lgbm"
    assert recipe.target_transform_name == "log1p"


def test_recipe_builder_contains_residual_recipe_in_default_recipes() -> None:
    """デフォルトレシピに複合モデル用レシピが含まれていることを確認する。"""
    recipe = RecipeBuilder().build("base_pls_snv_lgbm_residual")

    assert recipe.name == "base_pls_snv_lgbm_residual"
    assert recipe.preprocessor_name == "base_identity_pipeline"
    assert recipe.model_name == "base_pls_snv_lgbm_residual"
    assert recipe.target_transform_name == "identity"


def test_recipe_builder_contains_pls_oof_feature_recipe_in_default_recipes() -> None:
    """デフォルトレシピに PLS OOF 特徴量追加版が含まれていることを確認する。"""
    recipe = RecipeBuilder().build("snv_lgbm_with_raw_pls_oof")

    assert recipe.name == "snv_lgbm_with_raw_pls_oof"
    assert recipe.preprocessor_name == "snv_lgbm_with_raw_pls_oof_pipeline"
    assert recipe.model_name == "snv_lgbm"
