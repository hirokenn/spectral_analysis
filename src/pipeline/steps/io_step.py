from __future__ import annotations

from typing import Any

from src.data.dataset import Dataset
from src.model.mlflow_handler import MLflowHandler
from src.pipeline.pipeline import BaseStep
from src.pipeline.state import StateLike


class DataLoad(BaseStep):
    """CSV からデータを読み込み state.dataset に設定する。"""

    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        state.dataset = Dataset.from_csv(self.data_path)
        return state


class ModelSave(BaseStep):
    """学習済みモデルを mlflow に保存する。

    保存後、state.model_uri に保存先 URI を記録する。
    """

    def __init__(
        self,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state, require_non_none=["model", "model_type"])
        assert state.model is not None
        assert state.model_type is not None
        model_uri = MLflowHandler.log_model(
            model_type=state.model_type,
            native_model=state.model.get_native_model(),
            artifact_path=self.artifact_path,
            registered_model_name=self.registered_model_name,
        )
        state.model_uri = model_uri
        self.log_out = f"model_uri={model_uri}"
        return state


class ModelLoad(BaseStep):
    """mlflow からモデルを読み込み state.model に設定する。

    state.model に既にラッパーが存在する前提で、
    set_native_model() を通じてネイティブモデルを注入する。
    """

    def __init__(self, model_uri: str) -> None:
        super().__init__()
        self.model_uri = model_uri

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state, require_non_none=["model", "model_type"])
        assert state.model is not None
        assert state.model_type is not None
        native_model = MLflowHandler.load_model(
            model_type=state.model_type,
            model_uri=self.model_uri,
        )
        state.model.set_native_model(native_model)
        state.model_uri = self.model_uri
        self.log_out = f"model_uri={self.model_uri}"
        return state
