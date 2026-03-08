from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.model.base_model import ModelType
from src.pipeline.state import TrainState
from src.pipeline.steps.model_steps import (
    EvaluateOOF,
    PlotOOF,
    PredictTest,
    TrainCV,
    TrainFull,
)
from tests.helpers import (
    DummyMeanModel,
    DummyModelBuilder,
    make_dataset,
    make_recipe_builder,
    subset,
)


def test_train_cv_updates_oof_predictions() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        groups=np.array([1, 1, 2, 2]),
        sample_id=np.array([10, 11, 12, 13]),
    )
    cv = [
        (subset(ds, [0, 1]), subset(ds, [2, 3])),
        (subset(ds, [2, 3]), subset(ds, [0, 1])),
    ]
    state = TrainState(dataset=ds, model_type=ModelType.SKLEARN, model=DummyMeanModel())
    step = TrainCV(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        recipe_name="dummy_recipe",
    )

    updated = step.execute(state, cv=cv)

    assert updated.oof_predictions is not None
    expected = np.array([3.5, 3.5, 1.5, 1.5], dtype=np.float32)
    np.testing.assert_allclose(updated.oof_predictions, expected)


def test_train_cv_averages_when_sample_appears_multiple_times() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array([1, 1, 2]),
        sample_id=np.array([20, 21, 22]),
    )
    cv = [
        (subset(ds, [0, 1]), subset(ds, [2])),
        (subset(ds, [1, 2]), subset(ds, [0])),
        (subset(ds, [0, 2]), subset(ds, [1])),
    ]
    state = TrainState(dataset=ds, model_type=ModelType.SKLEARN, model=DummyMeanModel())
    step = TrainCV(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        recipe_name="dummy_recipe",
    )

    updated = step.execute(state, cv=cv)

    assert updated.oof_predictions is not None
    expected = np.array([2.5, 2.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(updated.oof_predictions, expected)


def test_train_full_sets_fitted_model_and_preprocessor() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array([1, 1, 2]),
        sample_id=np.array([1, 2, 3]),
    )
    state = TrainState(dataset=ds)
    step = TrainFull(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        recipe_name="dummy_recipe",
    )

    updated = step.execute(state)

    assert updated.preprocessor is not None
    assert updated.model is not None
    assert updated.model_type == ModelType.SKLEARN


def test_predict_test_updates_predictions() -> None:
    train_ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array([1, 1, 2]),
        sample_id=np.array([1, 2, 3]),
    )
    test_ds = make_dataset(
        X=np.array([[10.0], [11.0]]),
        y=None,
        groups=np.array([3, 3]),
        sample_id=np.array([100, 101]),
    )
    state = TrainState(train_dataset=train_ds, test_dataset=test_ds)
    train_step = TrainFull(
        recipe_builder=make_recipe_builder(),
        model_builder=DummyModelBuilder(),
        recipe_name="dummy_recipe",
    )
    predict_step = PredictTest()

    trained = train_step.execute(state)
    predicted = predict_step.execute(trained)

    assert predicted.test_predictions is not None
    np.testing.assert_allclose(
        predicted.test_predictions,
        np.array([2.0, 2.0], dtype=np.float32),
    )


def test_evaluate_oof_updates_rmse_metrics() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        groups=np.array([1, 1, 2, 2]),
        sample_id=np.array([1, 2, 3, 4]),
    )
    state = TrainState(dataset=ds, oof_predictions=np.array([1.0, 1.0, 4.0, 4.0]))
    step = EvaluateOOF()

    updated = step.execute(state)

    assert "overall_rmse" in updated.metrics
    assert "group_rmse_mean" in updated.metrics
    assert "group_rmse_by_group" in updated.metrics
    np.testing.assert_allclose(updated.metrics["overall_rmse"], np.sqrt(0.5))
    np.testing.assert_allclose(updated.metrics["group_rmse_mean"], np.sqrt(0.5))


@patch("src.pipeline.steps.model_steps.mlflow")
def test_plot_oof_logs_html_artifacts(mock_mlflow) -> None:  # type: ignore[no-untyped-def]
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array([1, 1, 2]),
        sample_id=np.array([1, 2, 3]),
    )
    state = TrainState(
        dataset=ds,
        oof_predictions=np.array([1.1, 2.2, 2.8]),
    )
    step = PlotOOF()
    step.execute(state)

    assert mock_mlflow.log_text.call_count == 2
    artifact_names = [call.args[1] for call in mock_mlflow.log_text.call_args_list]
    assert "plots/correlation.html" in artifact_names
    assert "plots/residual_histogram.html" in artifact_names


@patch("src.pipeline.steps.model_steps.mlflow")
def test_plot_oof_includes_feature_importance_when_model_set(mock_mlflow) -> None:  # type: ignore[no-untyped-def]
    class FeatureImportanceModel(DummyMeanModel):
        def feature_importance(self) -> dict[str, float] | None:
            return {"f1": 0.1, "f2": 0.2}

    ds = make_dataset(
        X=np.array([[0.0, 1.0], [1.0, 2.0]]),
        y=np.array([1.0, 2.0]),
        groups=np.array([1, 2]),
        sample_id=np.array([1, 2]),
    )
    model = FeatureImportanceModel()
    model.fit(ds)
    state = TrainState(
        dataset=ds,
        oof_predictions=np.array([1.0, 2.0]),
        model=model,
    )
    step = PlotOOF()
    step.execute(state)

    artifact_names = [call.args[1] for call in mock_mlflow.log_text.call_args_list]
    assert "plots/correlation.html" in artifact_names
    assert "plots/residual_histogram.html" in artifact_names
    assert "plots/feature_importance.html" in artifact_names
