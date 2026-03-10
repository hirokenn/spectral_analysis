from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.preprocess.core.base import BasePreprocessor
from src.preprocess.core.feature_union import FeatureUnion
from src.preprocess.core.pipeline import PreprocessingPipeline
from src.preprocess.features.dwt_features import DWTFeatureExtractor
from src.preprocess.features.group_sequence_features import (
    GroupSequenceFeatureExtractor,
)
from src.preprocess.features.interval_features import (
    IntervalMeanFeatureExtractor,
    IntervalSlopeFeatureExtractor,
)
from src.preprocess.features.water_band_summary import WaterBandSummaryFeatureExtractor
from src.preprocess.transforms.identity import IdentityPreprocessor
from src.preprocess.transforms.savgol import SavitzkyGolayPreprocessor
from src.preprocess.transforms.snv import SNVPreprocessor


class BuildType(StrEnum):
    """PreprocessorBuilder で扱う build_type 定義。"""

    IDENTITY = "identity"
    SNV = "snv"
    SAVITZKY_GOLAY = "savgol"
    GROUP_SEQUENCE = "group_sequence"
    INTERVAL_MEAN = "interval_mean"
    INTERVAL_SLOPE = "interval_slope"
    DWT = "dwt"
    WATER_BAND_SUMMARY = "water_band_summary"
    PIPELINE = "pipeline"
    FEATURE_UNION = "feature_union"

    @classmethod
    def from_value(cls, value: str) -> BuildType:
        """文字列を `BuildType` に変換する。"""
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(build_type.value for build_type in cls)
            raise ValueError(
                f"未対応の build_type です: {value} (allowed: {allowed})"
            ) from exc


class PreprocessorBuilder:
    """`preprocess_params.json` の定義から前処理インスタンスを構築する。"""

    def __init__(
        self,
        config_path: str | Path = "preprocess_params.json",
        config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """前処理定義を初期化する。"""
        if config is not None:
            self.config = config
            return

        with Path(config_path).open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise ValueError("前処理設定はオブジェクト形式である必要があります")
        self.config = loaded

    def build(self, preprocessor_name: str) -> BasePreprocessor:
        """前処理名を受け取り、定義に応じた前処理を返す。"""
        return self._build_recursive(preprocessor_name, ancestors=())

    def _build_recursive(
        self,
        preprocessor_name: str,
        ancestors: tuple[str, ...],
    ) -> BasePreprocessor:
        """前処理定義を再帰的に辿ってインスタンス化する。"""
        if preprocessor_name in ancestors:
            cycle = " -> ".join([*ancestors, preprocessor_name])
            raise ValueError(f"前処理定義が循環参照しています: {cycle}")

        preprocessor_config = self._get_preprocessor_config(preprocessor_name)
        build_type = BuildType.from_value(preprocessor_config["build_type"])

        if build_type == BuildType.PIPELINE:
            step_names = self._get_reference_names(preprocessor_config, key="steps")
            steps = [
                self._build_recursive(
                    step_name, ancestors=(*ancestors, preprocessor_name)
                )
                for step_name in step_names
            ]
            return PreprocessingPipeline(steps=steps)

        if build_type == BuildType.FEATURE_UNION:
            branch_names = self._get_reference_names(
                preprocessor_config, key="branches"
            )
            branches = [
                self._build_recursive(
                    branch_name, ancestors=(*ancestors, preprocessor_name)
                )
                for branch_name in branch_names
            ]
            return FeatureUnion(branches=branches)

        return self._build_single_preprocessor(preprocessor_name)

    def _build_single_preprocessor(self, preprocessor_name: str) -> BasePreprocessor:
        """単体前処理を build_type に応じて構築する。"""
        preprocessor_config = self._get_preprocessor_config(preprocessor_name)
        build_type = BuildType.from_value(preprocessor_config["build_type"])
        params = preprocessor_config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(
                f"params はオブジェクト形式である必要があります: {preprocessor_name}"
            )

        if build_type == BuildType.IDENTITY:
            return IdentityPreprocessor(**params)
        if build_type == BuildType.SNV:
            return SNVPreprocessor(**params)
        if build_type == BuildType.SAVITZKY_GOLAY:
            return SavitzkyGolayPreprocessor(**params)
        if build_type == BuildType.GROUP_SEQUENCE:
            return GroupSequenceFeatureExtractor(**params)
        if build_type == BuildType.INTERVAL_MEAN:
            return IntervalMeanFeatureExtractor(**params)
        if build_type == BuildType.INTERVAL_SLOPE:
            return IntervalSlopeFeatureExtractor(**params)
        if build_type == BuildType.DWT:
            return DWTFeatureExtractor(**params)
        if build_type == BuildType.WATER_BAND_SUMMARY:
            return WaterBandSummaryFeatureExtractor(**params)

        raise ValueError(
            f"未対応の build_type です: {build_type} (preprocessor={preprocessor_name})"
        )

    def _get_preprocessor_config(self, preprocessor_name: str) -> dict[str, Any]:
        """前処理名に対応する設定を取り出して検証する。"""
        if preprocessor_name not in self.config:
            raise ValueError(f"前処理定義が見つかりません: {preprocessor_name}")

        preprocessor_config = self.config[preprocessor_name]
        if not isinstance(preprocessor_config, dict):
            raise ValueError(f"前処理定義が不正です: {preprocessor_name}")
        if "build_type" not in preprocessor_config:
            raise ValueError(f"'build_type' が未指定です: {preprocessor_name}")

        return preprocessor_config

    def _get_reference_names(
        self,
        preprocessor_config: dict[str, Any],
        *,
        key: str,
    ) -> list[str]:
        """再帰参照する前処理名のリストを検証して返す。"""
        names = preprocessor_config.get(key)
        if not isinstance(names, list) or not names:
            raise ValueError(f"'{key}' は 1 件以上の配列で指定してください")
        if not all(isinstance(name, str) for name in names):
            raise ValueError(f"'{key}' の要素は文字列である必要があります")
        return names
