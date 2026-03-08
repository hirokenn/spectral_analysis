from __future__ import annotations

from typing import Any, List, Union

import gc
import mlflow  # type: ignore[import-untyped]

from src.pipeline.pipeline import BaseStep
from src.pipeline.state import StateLike


class SetNone(BaseStep):
    """指定属性を `None` にし、不要オブジェクトを解放する。"""

    def __init__(self, attribute_name: Union[str, List[str]]):
        super().__init__()
        self.attribute_name = attribute_name

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        if isinstance(self.attribute_name, list):
            for attr in self.attribute_name:
                setattr(state, attr, None)
        else:
            setattr(state, self.attribute_name, None)

        gc.collect()
        return state


class MlflowStartRun(BaseStep):
    """MLflow run を開始し、state.run_id に記録する。"""

    def __init__(
        self,
        *,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
        tracking_uri: str | None = "sqlite:///mlflow.db",
        experiment_name: str | None = "spectral_analysis",
    ) -> None:
        super().__init__()
        self.run_name = run_name
        self.nested = nested
        self.tags = tags
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name is not None:
            mlflow.set_experiment(self.experiment_name)

        run = mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
            tags=self.tags,
        )
        state.run_id = run.info.run_id
        self.log_out = f"run_id={state.run_id}"
        return state


class MlflowEndRun(BaseStep):
    """MLflow run を終了する。active run が無い場合は何もしない。"""

    def __init__(self, status: str = "FINISHED") -> None:
        super().__init__()
        self.status = status

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        active = mlflow.active_run()
        if active is None:
            self.log_out = "no_active_run"
            return state

        run_id = active.info.run_id
        mlflow.end_run(status=self.status)
        self.log_out = f"run_id={run_id}, status={self.status}"
        return state
