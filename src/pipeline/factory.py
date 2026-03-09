from __future__ import annotations

from src.model.model_builder import ModelBuilder
from src.pipeline.pipeline import Pipeline
from src.pipeline.steps.model_steps import (
    EvaluateOOF,
    PlotOOF,
    PredictTest,
    TrainCV,
    TrainFull,
)
from src.pipeline.steps.run_step import MlflowEndRun, MlflowStartRun
from src.preprocess.preprocessor_builder import PreprocessorBuilder
from src.recipes.builder import RecipeBuilder


def build_cv_pipeline(
    *,
    recipe_builder: RecipeBuilder,
    model_builder: ModelBuilder,
    preprocessor_builder: PreprocessorBuilder,
    recipe_name: str,
    run_name: str | None = None,
    tracking_uri: str | None = "sqlite:///mlflow.db",
    experiment_name: str | None = "spectral_analysis",
    plot_oof: bool = True,
    verbose: bool = True,
) -> Pipeline:
    """CV 学習・評価用パイプラインを返す。"""
    pipeline = (
        MlflowStartRun(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        >> TrainCV(
            recipe_builder=recipe_builder,
            model_builder=model_builder,
            preprocessor_builder=preprocessor_builder,
            recipe_name=recipe_name,
        )
        >> EvaluateOOF()
    )
    if plot_oof:
        pipeline = pipeline >> PlotOOF()
    pipeline = pipeline >> MlflowEndRun()
    pipeline.verbose = verbose
    return pipeline


def build_submission_pipeline(
    *,
    recipe_builder: RecipeBuilder,
    model_builder: ModelBuilder,
    preprocessor_builder: PreprocessorBuilder,
    recipe_name: str,
    run_name: str | None = None,
    tracking_uri: str | None = "sqlite:///mlflow.db",
    experiment_name: str | None = "spectral_analysis",
    verbose: bool = True,
) -> Pipeline:
    """全学習データで学習し、test データを予測する提出用パイプラインを返す。"""
    pipeline = (
        MlflowStartRun(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        >> TrainFull(
            recipe_builder=recipe_builder,
            model_builder=model_builder,
            preprocessor_builder=preprocessor_builder,
            recipe_name=recipe_name,
        )
        >> PredictTest()
        >> MlflowEndRun()
    )
    pipeline.verbose = verbose
    return pipeline
