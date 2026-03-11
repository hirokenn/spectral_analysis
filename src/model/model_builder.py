from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.model.base_model import BaseModel
from src.model.regressors.lightgbm import LightGBMRegressor
from src.model.regressors.pls import PLSRegressor
from src.model.regressors.residual_ensemble import ResidualEnsembleRegressor
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder


class BuildType(StrEnum):
    """ModelBuilder で扱う build_type 定義。"""

    PLS = "pls"
    LIGHTGBM = "lightgbm"
    RESIDUAL_ENSEMBLE = "residual_ensemble"
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
        preprocessor_builder: PreprocessorBuilder | None = None,
    ) -> None:
        """モデル定義を初期化する。"""
        self.preprocessor_builder = preprocessor_builder
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
        return self._build_recursive(model_name, ancestors=())

    def _build_recursive(
        self,
        model_name: str,
        *,
        ancestors: tuple[str, ...],
    ) -> BaseModel:
        """モデル定義を再帰的に辿ってインスタンス化する。"""
        if model_name in ancestors:
            cycle = " -> ".join([*ancestors, model_name])
            raise ValueError(f"モデル定義が循環参照しています: {cycle}")

        model_config = self._get_model_config(model_name)
        build_type = BuildType.from_value(model_config["build_type"])

        if build_type == BuildType.STACKING:
            raise NotImplementedError(
                "stacking の build は未実装です。"
                "将来は base_models を再帰的に build してください。"
            )
        if build_type == BuildType.RESIDUAL_ENSEMBLE:
            preprocessor_builder = self._resolve_preprocessor_builder()
            base_model_name = self._get_reference_name(
                model_config,
                key="base_model_name",
            )
            residual_model_name = self._get_reference_name(
                model_config,
                key="residual_model_name",
            )
            base_preprocessor_name = self._get_reference_name(
                model_config,
                key="base_preprocessor_name",
            )
            residual_preprocessor_name = self._get_reference_name(
                model_config,
                key="residual_preprocessor_name",
            )
            next_ancestors = (*ancestors, model_name)
            return ResidualEnsembleRegressor(
                base_preprocessor=preprocessor_builder.build(base_preprocessor_name),
                base_model=self._build_recursive(
                    base_model_name,
                    ancestors=next_ancestors,
                ),
                residual_preprocessor=preprocessor_builder.build(
                    residual_preprocessor_name
                ),
                residual_model=self._build_recursive(
                    residual_model_name,
                    ancestors=next_ancestors,
                ),
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
        return self._resolve_model_config(model_name, ancestors=())

    def _resolve_model_config(
        self,
        model_name: str,
        *,
        ancestors: tuple[str, ...],
    ) -> dict[str, Any]:
        """継承を解決したモデル設定を返す。"""
        if model_name in ancestors:
            cycle = " -> ".join([*ancestors, model_name])
            raise ValueError(f"モデル設定の継承が循環参照しています: {cycle}")
        if model_name not in self.config:
            raise ValueError(f"モデル定義が見つかりません: {model_name}")

        model_config = self.config[model_name]
        if not isinstance(model_config, dict):
            raise ValueError(f"モデル定義が不正です: {model_name}")
        parent_name = model_config.get("extends")
        if parent_name is None:
            resolved = dict(model_config)
        else:
            if not isinstance(parent_name, str) or not parent_name:
                raise ValueError(
                    f"'extends' は空でない文字列で指定してください: {model_name}"
                )
            parent_config = self._resolve_model_config(
                parent_name,
                ancestors=(*ancestors, model_name),
            )
            resolved = self._merge_model_config(parent_config, model_config)

        if "build_type" not in resolved:
            raise ValueError(f"'build_type' が未指定です: {model_name}")
        return resolved

    @staticmethod
    def _get_reference_name(model_config: dict[str, Any], *, key: str) -> str:
        """参照名を検証して返す。"""
        value = model_config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"'{key}' は空でない文字列で指定してください")
        return value

    def _resolve_preprocessor_builder(self) -> PreprocessorBuilder:
        """複合モデル構築に使う前処理 builder を返す。"""
        if self.preprocessor_builder is None:
            self.preprocessor_builder = PreprocessorBuilder()
        return self.preprocessor_builder

    @staticmethod
    def _merge_model_config(
        parent_config: dict[str, Any],
        child_config: dict[str, Any],
    ) -> dict[str, Any]:
        """親子のモデル設定をマージする。"""
        merged = dict(parent_config)
        child_without_extends = {
            key: value for key, value in child_config.items() if key != "extends"
        }
        parent_params = parent_config.get("params")
        child_params = child_without_extends.get("params")
        if isinstance(parent_params, dict) and isinstance(child_params, dict):
            child_without_extends["params"] = {**parent_params, **child_params}
        merged.update(child_without_extends)
        return merged
