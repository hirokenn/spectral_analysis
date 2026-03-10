from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import Dataset
from src.data.group_cv import LOSOCV
from src.model.regressors.pls import PLSRegressor
from src.preprocess.core.base import ComposablePreprocessor


@dataclass
class PLSOOFFeatureExtractor(ComposablePreprocessor):
    """PLS の OOF 予測を 1 列の特徴量として返す前処理。"""

    n_components: int = 8
    scale: bool = True
    feature_name: str = "raw_pls_oof_pred"
    fitted_wavenumbers_: np.ndarray | None = field(default=None, init=False)
    full_model_: PLSRegressor | None = field(default=None, init=False)

    def fit(self, ds: Dataset) -> None:
        """全学習データで PLS を学習し、transform 用モデルを保持する。"""
        self._fit_full_model(ds)

    def transform(self, ds: Dataset) -> Dataset:
        """fit 時に学習した PLS 予測を特徴量として返す。"""
        self._validate_input(ds)
        assert self.full_model_ is not None
        predictions = np.asarray(
            self.full_model_.predict(ds), dtype=np.float32
        ).reshape(-1, 1)
        return self._to_feature_dataset(ds, predictions)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """学習データでは PLS の OOF 予測を特徴量として返す。"""
        self._fit_full_model(ds)
        predictions = self._build_oof_predictions(ds)
        return self._to_feature_dataset(ds, predictions)

    def _fit_full_model(self, ds: Dataset) -> None:
        """transform 用に全学習データで PLS を学習する。"""
        if ds.y is None:
            raise ValueError("PLS OOF 特徴量には y が必要です")
        self.fitted_wavenumbers_ = ds.wavenumbers.copy()
        self.full_model_ = PLSRegressor(
            n_components=self.n_components,
            scale=self.scale,
        )
        self.full_model_.fit(ds)

    def _build_oof_predictions(self, ds: Dataset) -> np.ndarray:
        """LOSO CV で OOF 予測を構築する。"""
        if ds.y is None:
            raise ValueError("PLS OOF 特徴量には y が必要です")
        if ds.groups is None:
            raise ValueError("PLS OOF 特徴量には groups が必要です")

        sample_id_to_index = {
            int(sample_id): idx for idx, sample_id in enumerate(ds.sample_id)
        }
        predictions = np.full(ds.X.shape[0], np.nan, dtype=np.float32)

        for train_ds, valid_ds in LOSOCV(dataset=ds, show_progress=False):
            model = PLSRegressor(
                n_components=self.n_components,
                scale=self.scale,
            )
            model.fit(train_ds)
            valid_predictions = np.asarray(model.predict(valid_ds), dtype=np.float32)
            valid_idx = np.asarray(
                [
                    sample_id_to_index[int(sample_id)]
                    for sample_id in valid_ds.sample_id
                ],
                dtype=np.int64,
            )
            predictions[valid_idx] = valid_predictions

        if np.isnan(predictions).any():
            raise ValueError("PLS OOF 予測を全サンプルに割り当てできませんでした")
        return predictions.reshape(-1, 1)

    def _validate_input(self, ds: Dataset) -> None:
        """fit 時と同じ波数軸だけを受け付ける。"""
        if self.full_model_ is None or self.fitted_wavenumbers_ is None:
            raise ValueError("先に fit を実行してください")
        if not np.array_equal(ds.wavenumbers, self.fitted_wavenumbers_):
            raise ValueError("fit 時と異なる wavenumbers には変換できません")

    def _to_feature_dataset(self, ds: Dataset, predictions: np.ndarray) -> Dataset:
        """予測値を 1 列特徴量の Dataset に変換する。"""
        return ds.with_features(
            predictions,
            wavenumbers=np.array([np.nan], dtype=np.float32),
            feature_names=[self.feature_name],
        )
