from __future__ import annotations

from pathlib import Path

import mlflow  # type: ignore[import-untyped]
import numpy as np

from src.pipeline.factory import build_cv_pipeline, build_submission_pipeline
from src.pipeline.state import TrainState
from tests.helpers import (
    DummyModelBuilder,
    DummyPreprocessorBuilder,
    make_dataset,
    make_recipe_builder,
)


def test_build_cv_pipeline_has_expected_steps() -> None:
    pipeline = build_cv_pipeline(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        preprocessor_builder=DummyPreprocessorBuilder(),
        recipe_name="dummy_recipe",
        verbose=False,
    )

    assert [step.__class__.__name__ for step in pipeline.steps] == [
        "MlflowStartRun",
        "TrainCV",
        "EvaluateOOF",
        "PlotOOF",
        "MlflowEndRun",
    ]


def test_build_submission_pipeline_has_expected_steps() -> None:
    pipeline = build_submission_pipeline(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        preprocessor_builder=DummyPreprocessorBuilder(),
        recipe_name="dummy_recipe",
        verbose=False,
    )

    assert [step.__class__.__name__ for step in pipeline.steps] == [
        "MlflowStartRun",
        "TrainFull",
        "PredictTest",
        "MlflowEndRun",
    ]


def test_cv_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    tracking_uri = f"file://{tmp_path}/mlruns"
    dataset = make_dataset(
        X=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        y=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        groups=np.array([1, 2, 3, 4], dtype=np.int64),
        sample_id=np.array([10, 11, 12, 13], dtype=np.int64),
    )
    state = TrainState(dataset=dataset)
    pipeline = build_cv_pipeline(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        preprocessor_builder=DummyPreprocessorBuilder(),
        recipe_name="dummy_recipe",
        tracking_uri=tracking_uri,
        experiment_name="test_cv_pipeline",
        verbose=False,
    )

    updated = pipeline.run(state)

    assert updated.run_id is not None
    assert updated.oof_predictions is not None
    assert "overall_rmse" in updated.metrics
    assert "group_rmse_mean" in updated.metrics
    assert mlflow.active_run() is None


def test_submission_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    tracking_uri = f"file://{tmp_path}/mlruns"
    train_ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        groups=np.array([1, 1, 2], dtype=np.int64),
        sample_id=np.array([1, 2, 3], dtype=np.int64),
    )
    test_ds = make_dataset(
        X=np.array([[10.0], [11.0]], dtype=np.float32),
        y=None,
        groups=np.array([3, 3], dtype=np.int64),
        sample_id=np.array([100, 101], dtype=np.int64),
    )
    state = TrainState(train_dataset=train_ds, test_dataset=test_ds)
    pipeline = build_submission_pipeline(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        preprocessor_builder=DummyPreprocessorBuilder(),
        recipe_name="dummy_recipe",
        tracking_uri=tracking_uri,
        experiment_name="test_submission_pipeline",
        verbose=False,
    )

    updated = pipeline.run(state)

    assert updated.run_id is not None
    assert updated.model is not None
    assert updated.preprocessor is not None
    assert updated.test_predictions is not None
    np.testing.assert_allclose(
        updated.test_predictions,
        np.array([2.0, 2.0], dtype=np.float32),
    )
    assert mlflow.active_run() is None
