from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from src.model.base_model import ModelType

_FLAVOR_MODULE: dict[ModelType, str] = {
    ModelType.SKLEARN: "mlflow.sklearn",
    ModelType.LIGHTGBM: "mlflow.lightgbm",
    ModelType.CATBOOST: "mlflow.catboost",
    ModelType.PYTORCH: "mlflow.pytorch",
}


class MLflowHandler:
    """ModelType に応じた mlflow flavor の保存・読み込みを一元管理する。

    新しい ModelType を追加するときは `_FLAVOR_MODULE` にマッピングを足すだけで対応できる。
    """

    @staticmethod
    def _get_flavor(model_type: ModelType) -> ModuleType:
        """ModelType に対応する mlflow flavor モジュールを返す。"""
        module_name = _FLAVOR_MODULE.get(model_type)
        if module_name is None:
            allowed = ", ".join(mt.value for mt in _FLAVOR_MODULE)
            raise ValueError(
                f"未対応の model_type です: {model_type} (allowed: {allowed})"
            )
        return importlib.import_module(module_name)

    @classmethod
    def log_model(
        cls,
        model_type: ModelType,
        native_model: Any,
        *,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """ネイティブモデルを mlflow に保存し、model_uri を返す。"""
        flavor = cls._get_flavor(model_type)
        log_kwargs: dict[str, Any] = {"artifact_path": artifact_path}
        if registered_model_name is not None:
            log_kwargs["registered_model_name"] = registered_model_name
        model_info = flavor.log_model(native_model, **log_kwargs)
        return str(model_info.model_uri)

    @classmethod
    def load_model(cls, model_type: ModelType, model_uri: str) -> Any:
        """model_uri からネイティブモデルを読み込む。"""
        flavor = cls._get_flavor(model_type)
        return flavor.load_model(model_uri)
