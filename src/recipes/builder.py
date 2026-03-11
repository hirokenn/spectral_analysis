from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.recipes.base import Recipe
from src.recipes.recipes import RECIPES


@dataclass
class RecipeBuilder:
    """Recipe 名から定義を取得するビルダー。"""

    recipes: Mapping[str, Recipe] = field(default_factory=lambda: RECIPES)

    def build(self, recipe_name: str) -> Recipe:
        """Recipe 名に対応する定義を返す。"""
        if recipe_name not in self.recipes:
            raise ValueError(f"レシピ定義が見つかりません: {recipe_name}")
        return self.recipes[recipe_name]
