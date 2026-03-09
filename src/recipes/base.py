from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    """前処理とモデル条件をまとめた実行レシピ。"""

    name: str
    preprocessor_name: str
    model_name: str
