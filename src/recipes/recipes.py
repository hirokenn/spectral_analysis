from __future__ import annotations

from src.recipes.base import Recipe


def _make_recipe(
    name: str,
    *,
    preprocessor_name: str,
    model_name: str | None = None,
    target_transform_name: str = "identity",
) -> Recipe:
    """命名規則に沿った Recipe を生成する。"""
    return Recipe(
        name=name,
        preprocessor_name=preprocessor_name,
        model_name=name if model_name is None else model_name,
        target_transform_name=target_transform_name,
    )


def _register_standard_recipe(
    recipes: dict[str, Recipe],
    *,
    name: str,
    preprocessor_name: str,
    model_name: str | None = None,
) -> None:
    """通常レシピを登録する。"""
    recipes[name] = _make_recipe(
        name,
        preprocessor_name=preprocessor_name,
        model_name=model_name,
    )


def _build_default_recipes() -> dict[str, Recipe]:
    """デフォルト recipe 一覧を構築する。"""
    recipes: dict[str, Recipe] = {}
    _register_standard_recipe(
        recipes,
        name="base_pls",
        preprocessor_name="base_identity_pipeline",
    )
    _register_standard_recipe(
        recipes,
        name="snv_pls",
        preprocessor_name="snv_pls_pipeline",
    )
    _register_standard_recipe(
        recipes,
        name="snv_lgbm",
        preprocessor_name="snv_lgbm_pipeline",
    )
    _register_standard_recipe(
        recipes,
        name="snv_lgbm_with_raw_pls_oof",
        preprocessor_name="snv_lgbm_with_raw_pls_oof_pipeline",
        model_name="snv_lgbm",
    )
    recipes["snv_lgbm_log1p"] = _make_recipe(
        "snv_lgbm_log1p",
        preprocessor_name="snv_lgbm_pipeline",
        model_name="snv_lgbm",
        target_transform_name="log1p",
    )
    _register_standard_recipe(
        recipes,
        name="savgol_lgbm",
        preprocessor_name="savgol_lgbm_pipeline",
    )
    recipes["savgol_lgbm_log1p"] = _make_recipe(
        "savgol_lgbm_log1p",
        preprocessor_name="savgol_lgbm_pipeline",
        model_name="savgol_lgbm",
        target_transform_name="log1p",
    )
    recipes["base_pls_snv_lgbm_residual"] = _make_recipe(
        "base_pls_snv_lgbm_residual",
        preprocessor_name="base_identity_pipeline",
    )
    return recipes


RECIPES: dict[str, Recipe] = _build_default_recipes()
