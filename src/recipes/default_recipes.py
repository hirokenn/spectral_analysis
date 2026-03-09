from __future__ import annotations

from src.recipes.base import Recipe

RECIPES: dict[str, Recipe] = {
    "base_pls": Recipe(
        name="base_pls",
        preprocessor_name="base_identity_pipeline",
        model_name="base_pls",
    ),
    "snv_lgbm_feature_only": Recipe(
        name="snv_lgbm_feature_only",
        preprocessor_name="snv_lgbm_feature_only_pipeline",
        model_name="snv_lgbm_feature_only",
    ),
}
