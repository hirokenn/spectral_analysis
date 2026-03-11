from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BaseTargetTransformer(Protocol):
    """回帰ターゲット変換の共通インターフェース。"""

    def fit(self, y: np.ndarray) -> BaseTargetTransformer:
        """学習データから内部状態を初期化する。"""
        ...

    def transform(self, y: np.ndarray) -> np.ndarray:
        """目的変数を変換する。"""
        ...

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """変換済み目的変数を元スケールへ戻す。"""
        ...


@dataclass
class IdentityTargetTransformer:
    """目的変数をそのまま扱う恒等変換。"""

    def fit(self, y: np.ndarray) -> IdentityTargetTransformer:
        """恒等変換なので学習は不要。"""
        _ = y
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """入力をそのまま返す。"""
        return np.asarray(y, dtype=np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """入力をそのまま返す。"""
        return np.asarray(y, dtype=np.float32)


@dataclass
class Log1pTargetTransformer:
    """`log1p` と `expm1` による目的変数変換。"""

    def fit(self, y: np.ndarray) -> Log1pTargetTransformer:
        """`log1p` 適用可能な範囲かを検証する。"""
        numeric = np.asarray(y, dtype=np.float32)
        if np.any(numeric <= -1.0):
            raise ValueError("log1p 変換には y > -1 が必要です")
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """目的変数に `log1p` を適用する。"""
        numeric = np.asarray(y, dtype=np.float32)
        if np.any(numeric <= -1.0):
            raise ValueError("log1p 変換には y > -1 が必要です")
        return np.log1p(numeric).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """変換済み目的変数に `expm1` を適用する。"""
        return np.expm1(np.asarray(y, dtype=np.float32)).astype(np.float32)


def build_target_transformer(name: str) -> BaseTargetTransformer:
    """変換名に応じたターゲット変換器を返す。"""
    if name == "identity":
        return IdentityTargetTransformer()
    if name == "log1p":
        return Log1pTargetTransformer()
    raise ValueError(f"未対応の target_transform_name です: {name}")
