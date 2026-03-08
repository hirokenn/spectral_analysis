from __future__ import annotations

from typing import Any, Iterable, cast

import mlflow  # type: ignore[import-untyped]

from src.data.dataset import Dataset
from src.data.group_cv import GroupCV
from src.model.eval import OOFEvaluator
from src.model.model_builder import ModelBuilder
from src.model.plot import OOFPlotter
from src.model.trainers.cv_trainer import CVTrainer
from src.model.trainers.full_trainer import FullTrainer
from src.pipeline.pipeline import BaseStep
from src.pipeline.state import PredictState, StateLike, TrainState
from src.recipes.builder import RecipeBuilder


class TrainCV(BaseStep):
    """CV 学習器の実行を担当するラッパー Step。"""

    def __init__(
        self,
        recipe_builder: RecipeBuilder,
        model_builder: ModelBuilder,
        recipe_name: str,
    ) -> None:
        super().__init__()
        self.recipe_builder = recipe_builder
        self.model_builder = model_builder
        self.recipe_name = recipe_name

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """CVTrainer を実行し、結果を state に反映する。"""
        self._require_state(
            state,
            allowed_types=(TrainState,),
        )

        train_state = cast(TrainState, state)
        dataset = self._resolve_train_dataset(train_state)
        recipe = self.recipe_builder.build(self.recipe_name)
        cv = cast(
            Iterable[tuple[Dataset, Dataset]],
            kwargs.get("cv", GroupCV(dataset=dataset)),
        )
        trainer = CVTrainer(cv=cv, recipe=recipe, model_builder=self.model_builder)
        result = trainer.run(dataset=dataset)
        train_state.oof_predictions = result.oof_predictions
        train_state.recipe_name = recipe.name
        train_state.model_type = self.model_builder.build(recipe.model_name).model_type
        self.log_out = (
            f"recipe={recipe.name}, "
            f"splits={result.n_splits}, "
            f"covered={result.covered_count}/{result.sample_count}"
        )
        return train_state

    def _resolve_train_dataset(self, state: TrainState) -> Dataset:
        """学習対象データセットを解決する。"""
        if state.train_dataset is not None:
            return state.train_dataset
        if state.dataset is not None:
            return state.dataset
        raise ValueError("train_dataset も dataset も設定されていません")


class TrainFull(BaseStep):
    """全学習データを使って前処理とモデルを学習するラッパー Step。"""

    def __init__(
        self,
        recipe_builder: RecipeBuilder,
        model_builder: ModelBuilder,
        recipe_name: str,
    ) -> None:
        super().__init__()
        self.recipe_builder = recipe_builder
        self.model_builder = model_builder
        self.recipe_name = recipe_name

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """FullTrainer を実行し、学習済み前処理器とモデルを state に反映する。"""
        self._require_state(
            state,
            allowed_types=(TrainState,),
        )
        train_state = cast(TrainState, state)
        dataset = self._resolve_train_dataset(train_state)
        recipe = self.recipe_builder.build(self.recipe_name)
        trainer = FullTrainer(recipe=recipe, model_builder=self.model_builder)
        result = trainer.run(dataset)

        train_state.recipe_name = recipe.name
        train_state.preprocessor = result.preprocessor
        train_state.model = result.model
        train_state.model_type = result.model.model_type
        self.log_out = f"recipe={recipe.name}"
        return train_state

    def _resolve_train_dataset(self, state: TrainState) -> Dataset:
        """学習対象データセットを解決する。"""
        if state.train_dataset is not None:
            return state.train_dataset
        if state.dataset is not None:
            return state.dataset
        raise ValueError("train_dataset も dataset も設定されていません")


class PredictTest(BaseStep):
    """学習済み前処理器とモデルで test データの予測を作るラッパー Step。"""

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """前処理と推論を実行し、`state.test_predictions` を更新する。"""
        self._require_state(
            state,
            allowed_types=(TrainState, PredictState),
            require_non_none=["model", "preprocessor"],
        )
        dataset = self._resolve_test_dataset(state)
        assert state.preprocessor is not None
        assert state.model is not None

        processed_dataset = state.preprocessor.transform(dataset)
        state.test_predictions = state.model.predict(processed_dataset)
        self.log_out = f"n_predictions={state.test_predictions.shape[0]}"
        return state

    def _resolve_test_dataset(self, state: StateLike) -> Dataset:
        """予測対象データセットを解決する。"""
        if state.test_dataset is not None:
            return state.test_dataset
        if state.dataset is not None:
            return state.dataset
        raise ValueError("test_dataset も dataset も設定されていません")


class EvaluateOOF(BaseStep):
    """OOF 予測を評価し、TrainState に RMSE 指標を保存する。"""

    def __init__(self, evaluator: OOFEvaluator | None = None) -> None:
        super().__init__()
        self.evaluator = evaluator or OOFEvaluator()

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """全体 RMSE とグループ別 RMSE 平均を計算して state に反映する。"""
        self._require_state(
            state,
            allowed_types=(TrainState,),
            require_non_none=["oof_predictions"],
        )
        train_state = cast(TrainState, state)
        dataset = self._resolve_train_dataset(train_state)
        assert train_state.oof_predictions is not None

        result = self.evaluator.evaluate(train_state.oof_predictions, dataset)
        train_state.metrics["overall_rmse"] = result.overall_rmse
        train_state.metrics["group_rmse_mean"] = result.group_rmse_mean
        train_state.metrics["group_rmse_by_group"] = result.group_rmse_by_group
        self.log_out = (
            f"overall_rmse={result.overall_rmse:.6f}, "
            f"group_rmse_mean={result.group_rmse_mean:.6f}"
        )
        return train_state

    def _resolve_train_dataset(self, state: TrainState) -> Dataset:
        """評価対象の学習データセットを解決する。"""
        if state.train_dataset is not None:
            return state.train_dataset
        if state.dataset is not None:
            return state.dataset
        raise ValueError("train_dataset も dataset も設定されていません")


class PlotOOF(BaseStep):
    """OOF 予測結果のプロットを作成して MLflow に HTML として保存する。"""

    def __init__(self, plotter: OOFPlotter | None = None) -> None:
        super().__init__()
        self.plotter = plotter or OOFPlotter()

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """プロットを生成し、アクティブな MLflow run に artifact としてログする。"""
        self._require_state(
            state,
            allowed_types=(TrainState,),
            require_non_none=["oof_predictions"],
        )
        train_state = cast(TrainState, state)
        dataset = self._resolve_train_dataset(train_state)
        assert train_state.oof_predictions is not None

        figures = self.plotter.create_all(
            oof_predictions=train_state.oof_predictions,
            dataset=dataset,
            model=train_state.model,
        )

        for artifact_name, fig in figures.items():
            mlflow.log_text(fig.to_html(), artifact_name)

        self.log_out = f"logged={len(figures)} plots"
        return train_state

    def _resolve_train_dataset(self, state: TrainState) -> Dataset:
        """プロット対象の学習データセットを解決する。"""
        if state.train_dataset is not None:
            return state.train_dataset
        if state.dataset is not None:
            return state.dataset
        raise ValueError("train_dataset も dataset も設定されていません")
