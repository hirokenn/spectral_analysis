from __future__ import annotations

from pathlib import Path

import mlflow  # type: ignore[import-untyped]
import numpy as np

from src.model.base_model import ModelType
from src.model.regressors.pls import PLSRegressor
from src.pipeline.state import TrainState
from src.pipeline.steps.run_step import MlflowEndRun, MlflowStartRun


def make_state() -> TrainState:
    """run step テスト用の最小 state を作成する。"""
    from src.data.dataset import Dataset

    ds = Dataset(
        X=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        y=np.array([1.0, 2.0], dtype=np.float32),
        groups=np.array([1, 1], dtype=np.int64),
        sample_id=np.array([1, 2], dtype=np.int64),
        wavenumbers=np.array([10000.0, 9990.0], dtype=np.float32),
    )
    return TrainState(dataset=ds, model_type=ModelType.SKLEARN, model=PLSRegressor())


def test_mlflow_start_run_sets_run_id(tmp_path: Path) -> None:
    tracking_uri = f"file://{tmp_path}/mlruns"
    step = MlflowStartRun(
        run_name="unit_test_run",
        tags={"stage": "test"},
        tracking_uri=tracking_uri,
        experiment_name="unit_test_experiment",
    )
    state = make_state()

    updated = step.execute(state)

    assert updated.run_id is not None
    assert mlflow.active_run() is not None
    MlflowEndRun().execute(updated)


def test_mlflow_end_run_finishes_active_run(tmp_path: Path) -> None:
    tracking_uri = f"file://{tmp_path}/mlruns"
    state = make_state()
    start = MlflowStartRun(tracking_uri=tracking_uri, experiment_name="end_run_test")
    end = MlflowEndRun()

    started = start.execute(state)
    ended = end.execute(started)

    assert ended.run_id is not None
    assert mlflow.active_run() is None


def test_mlflow_end_run_without_active_run_is_noop() -> None:
    state = make_state()
    step = MlflowEndRun()

    updated = step.execute(state)

    assert updated is state
    assert step.log_out == "no_active_run"
