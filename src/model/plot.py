from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from src.data.dataset import Dataset


class OOFPlotter:
    """OOF 予測結果から plotly Figure を生成する。

    * correlation_plot: 予測 vs 実測 散布図
    * residual_histogram: 残差ヒストグラム
    * feature_importance_plot: 回帰係数 / feature_importances_（対応属性がある場合）
    * create_all: 上記をまとめて dict[artifact_name, Figure] で返す
    """

    def create_all(
        self,
        oof_predictions: np.ndarray,
        dataset: Dataset,
        model: Any = None,
    ) -> dict[str, go.Figure]:
        """全プロットを生成して {artifact ファイル名: Figure} を返す。"""
        if dataset.y is None:
            raise ValueError("プロットには dataset.y が必要です")

        valid_mask = ~np.isnan(oof_predictions)
        if not np.any(valid_mask):
            raise ValueError("有効な OOF 予測がありません")

        y_true = dataset.y[valid_mask]
        y_pred = oof_predictions[valid_mask]
        groups = dataset.groups[valid_mask] if dataset.groups is not None else None

        figures: dict[str, go.Figure] = {
            "plots/correlation.html": self.correlation_plot(y_true, y_pred, groups),
            "plots/residual_histogram.html": self.residual_histogram(y_true, y_pred),
        }

        if model is not None:
            fig_imp = self.feature_importance_plot(model)
            if fig_imp is not None:
                figures["plots/feature_importance.html"] = fig_imp

        return figures

    def correlation_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None = None
    ) -> go.Figure:
        """予測 vs 実測の散布図を返す。groups があれば色分けする。"""
        fig = go.Figure()
        if groups is None:
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    marker=dict(size=5, opacity=0.6),
                    name="OOF",
                )
            )
        else:
            for group_id in np.unique(groups):
                group_mask = groups == group_id
                fig.add_trace(
                    go.Scatter(
                        x=y_true[group_mask],
                        y=y_pred[group_mask],
                        mode="markers",
                        marker=dict(size=5, opacity=0.6),
                        name=f"group={int(group_id)}",
                    )
                )
        vmin = min(float(y_true.min()), float(y_pred.min()))
        vmax = max(float(y_true.max()), float(y_pred.max()))
        fig.add_trace(
            go.Scatter(
                x=[vmin, vmax],
                y=[vmin, vmax],
                mode="lines",
                line=dict(dash="dash", color="red"),
                name="y = x",
            )
        )
        fig.update_layout(
            title="OOF: Predicted vs Actual",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            template="plotly_white",
        )
        return fig

    def residual_histogram(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """残差のヒストグラムを返す。"""
        residuals = y_pred - y_true
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=residuals, nbinsx=50, name="Residuals"))
        fig.update_layout(
            title="OOF: Residual Distribution",
            xaxis_title="Residual (Predicted − Actual)",
            yaxis_title="Count",
            template="plotly_white",
        )
        return fig

    def feature_importance_plot(self, model: Any) -> go.Figure | None:
        """モデルが返す特徴量重要度辞書を棒グラフで返す。"""
        if not hasattr(model, "feature_importance"):
            return None

        feature_importance = model.feature_importance()
        if not feature_importance:
            return None

        feature_names = list(feature_importance.keys())
        importance = list(feature_importance.values())

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=importance,
                y=feature_names,
                orientation="h",
                name="Feature Importance",
            )
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
        )
        return fig
