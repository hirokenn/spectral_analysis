from __future__ import annotations

from dataclasses import dataclass

from src.data.dataset import Dataset
from src.model.base_model import BaseModel
from src.model.model_builder import ModelBuilder
from src.preprocess.base import BasePreprocessor
from src.recipes.base import Recipe


@dataclass
class TrainFullResult:
    """全件学習結果のコンテナ。"""

    preprocessor: BasePreprocessor
    model: BaseModel


@dataclass
class FullTrainer:
    """全学習データで前処理とモデル学習を行う学習器。"""

    recipe: Recipe
    model_builder: ModelBuilder

    def run(self, dataset: Dataset) -> TrainFullResult:
        """全件学習済みの前処理器とモデルを返す。"""
        if dataset.y is None:
            raise ValueError("TrainFull には y が必要です")

        preprocessor = self.recipe.preprocessor_factory()
        model = self.model_builder.build(self.recipe.model_name)
        processed_dataset = preprocessor.fit_transform(dataset)
        model.fit(processed_dataset)
        return TrainFullResult(preprocessor=preprocessor, model=model)
