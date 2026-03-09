from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.model.base_model import BaseModel
from src.model.regressors.lightgbm import LightGBMRegressor
from src.model.regressors.pls import PLSRegressor


class BuildType(StrEnum):
    """ModelBuilder で扱う build_type 定義。"""

    PLS = "pls"
    LIGHTGBM = "lightgbm"
    STACKING = "stacking"

    @classmethod
    def from_value(cls, value: str) -> BuildType:
        """文字列を `BuildType` に変換する。"""
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(bt.value for bt in cls)
            raise ValueError(
                f"未対応の build_type です: {value} (allowed: {allowed})"
            ) from exc


class ModelBuilder:
    """`params.json` の定義からモデルインスタンスを構築する。"""

    def __init__(
        self,
        config_path: str | Path = "params.json",
        config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """モデル定義を初期化する。"""
        if config is not None:
            self.config = config
            return

        with Path(config_path).open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise ValueError("モデル設定はオブジェクト形式である必要があります")
        self.config = loaded

    def build(self, model_name: str) -> BaseModel:
        """モデル名を受け取り、定義に応じたモデルを返す。"""
        model_config = self._get_model_config(model_name)
        build_type = BuildType.from_value(model_config["build_type"])

        if build_type == BuildType.STACKING:
            raise NotImplementedError(
                "stacking の build は未実装です。"
                "将来は base_models を再帰的に build してください。"
            )

        return self._build_single_model(model_name)

    def _build_single_model(self, model_name: str) -> BaseModel:
        """単体モデルを build_type に応じて構築する。"""
        model_config = self._get_model_config(model_name)
        build_type = BuildType.from_value(model_config["build_type"])
        params = model_config.get("params", {})

        if build_type == BuildType.PLS:
            return PLSRegressor(**params)
        if build_type == BuildType.LIGHTGBM:
            return LightGBMRegressor(**params)

        raise ValueError(f"未対応の build_type です: {build_type} (model={model_name})")

    def _get_model_config(self, model_name: str) -> dict[str, Any]:
        """モデル名に対応する設定を取り出して検証する。"""
        if model_name not in self.config:
            raise ValueError(f"モデル定義が見つかりません: {model_name}")

        model_config = self.config[model_name]
        if not isinstance(model_config, dict):
            raise ValueError(f"モデル定義が不正です: {model_name}")
        if "build_type" not in model_config:
            raise ValueError(f"'build_type' が未指定です: {model_name}")

        return model_config
