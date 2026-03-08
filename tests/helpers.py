from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import Dataset
from src.model.base_model import ModelType
from src.model.model_builder import ModelBuilder
from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.pipeline import PreprocessingPipeline
from src.recipes.base import Recipe
from src.recipes.builder import RecipeBuilder


@dataclass
class DummyMeanModel:
    """学習データの平均を返すテスト用モデル。"""

    mean_: float = 0.0

    @property
    def model_type(self) -> ModelType:
        return ModelType.SKLEARN

    def fit(self, ds: Dataset) -> None:
        assert ds.y is not None
        self.mean_ = float(np.mean(ds.y))

    def predict(self, ds: Dataset) -> np.ndarray:
        return np.full(ds.X.shape[0], self.mean_, dtype=np.float32)

    def get_native_model(self) -> Any:
        return self

    def set_native_model(self, native_model: Any) -> None:
        self.mean_ = float(native_model.mean_)


class DummyModelBuilder(ModelBuilder):
    """常に `DummyMeanModel` を返すテスト用 model builder。"""

    def __init__(self) -> None:
        super().__init__(config={"dummy_model": {"build_type": "pls", "params": {}}})

    def build(self, model_name: str) -> DummyMeanModel:
        _ = model_name
        return DummyMeanModel()


def make_recipe(recipe_name: str = "dummy_recipe") -> Recipe:
    """Identity 前処理を使うテスト用 recipe を返す。"""
    return Recipe(
        name=recipe_name,
        preprocessor_factory=lambda: PreprocessingPipeline([IdentityPreprocessor()]),
        model_name="dummy_model",
    )


def make_recipe_builder(recipe_name: str = "dummy_recipe") -> RecipeBuilder:
    """テスト用 recipe builder を返す。"""
    recipe = make_recipe(recipe_name)
    return RecipeBuilder(recipes={recipe.name: recipe})


def make_dataset(
    X: np.ndarray,
    y: np.ndarray | None,
    sample_id: np.ndarray,
    groups: np.ndarray | None = None,
) -> Dataset:
    """テスト用 `Dataset` を構築する。"""
    return Dataset(
        X=X.astype(np.float32),
        y=None if y is None else y.astype(np.float32),
        groups=None if groups is None else groups.astype(np.int64),
        sample_id=sample_id.astype(np.int64),
        wavenumbers=np.arange(X.shape[1], dtype=np.float32),
    )


def subset(ds: Dataset, indices: list[int]) -> Dataset:
    """指定行だけを抜き出した `Dataset` を返す。"""
    idx = np.asarray(indices, dtype=np.int64)
    return Dataset(
        X=ds.X[idx],
        y=None if ds.y is None else ds.y[idx],
        groups=None if ds.groups is None else ds.groups[idx],
        sample_id=ds.sample_id[idx],
        wavenumbers=ds.wavenumbers.copy(),
    )
