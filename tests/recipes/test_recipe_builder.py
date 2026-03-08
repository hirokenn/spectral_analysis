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
