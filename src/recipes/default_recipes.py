from __future__ import annotations

from src.recipes.base import Recipe

RECIPES: dict[str, Recipe] = {
    "base_pls": Recipe(
        name="base_pls",
        preprocessor_name="base_identity_pipeline",
        model_name="base_pls",
    ),
    "base_pls_log1p": Recipe(
        name="base_pls_log1p",
        preprocessor_name="base_identity_pipeline",
        model_name="base_pls",
        target_transform_name="log1p",
    ),
    "snv_pls": Recipe(
        name="snv_pls",
        preprocessor_name="snv_pls_pipeline",
        model_name="snv_pls",
    ),
    "snv_pls_log1p": Recipe(
        name="snv_pls_log1p",
        preprocessor_name="snv_pls_pipeline",
        model_name="snv_pls",
        target_transform_name="log1p",
    ),
    "snv_lgbm": Recipe(
        name="snv_lgbm",
        preprocessor_name="snv_lgbm_pipeline",
        model_name="snv_lgbm",
    ),
    "snv_lgbm_log1p": Recipe(
        name="snv_lgbm_log1p",
        preprocessor_name="snv_lgbm_pipeline",
        model_name="snv_lgbm",
        target_transform_name="log1p",
    ),
    "savgol_lgbm": Recipe(
        name="savgol_lgbm",
        preprocessor_name="savgol_lgbm_pipeline",
        model_name="savgol_lgbm",
    ),
    "savgol_lgbm_log1p": Recipe(
        name="savgol_lgbm_log1p",
        preprocessor_name="savgol_lgbm_pipeline",
        model_name="savgol_lgbm",
        target_transform_name="log1p",
    ),
}
