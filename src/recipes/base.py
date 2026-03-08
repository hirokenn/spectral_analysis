from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.preprocess.base import BasePreprocessor


@dataclass(frozen=True)
class Recipe:
    """前処理とモデル条件をまとめた実行レシピ。"""

    name: str
    preprocessor_factory: Callable[[], BasePreprocessor]
    model_name: str
