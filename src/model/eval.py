from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset import Dataset


@dataclass
class EvalResult:
    """OOF 評価結果のコンテナ。"""

    overall_rmse: float
    group_rmse_mean: float
    group_rmse_by_group: dict[int, float]


class OOFEvaluator:
    """OOF 予測と正解データから RMSE 指標を計算する。"""

    def evaluate(self, oof_predictions: np.ndarray, dataset: Dataset) -> EvalResult:
        """全体 RMSE とグループ別 RMSE 平均を返す。"""
        if dataset.y is None:
            raise ValueError("評価には dataset.y が必要です")
        if dataset.groups is None:
            raise ValueError("評価には dataset.groups が必要です")
        if oof_predictions.shape[0] != dataset.y.shape[0]:
            raise ValueError("oof_predictions と dataset.y の長さが一致しません")

        y = dataset.y
        groups = dataset.groups
        valid_mask = ~np.isnan(oof_predictions)
        if not np.any(valid_mask):
            raise ValueError("有効な OOF 予測がありません")

        overall_rmse = self._rmse(y[valid_mask], oof_predictions[valid_mask])

        group_rmse_by_group: dict[int, float] = {}
        for group_id in np.unique(groups[valid_mask]):
            group_mask = valid_mask & (groups == group_id)
            group_rmse_by_group[int(group_id)] = self._rmse(
                y[group_mask], oof_predictions[group_mask]
            )

        group_rmse_mean = float(np.mean(list(group_rmse_by_group.values())))
        return EvalResult(
            overall_rmse=overall_rmse,
            group_rmse_mean=group_rmse_mean,
            group_rmse_by_group=group_rmse_by_group,
        )

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE を計算する。"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
