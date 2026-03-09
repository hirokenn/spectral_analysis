from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.data.dataset import Dataset
from src.model.model_builder import ModelBuilder
from src.preprocess.preprocessor_builder import PreprocessorBuilder
from src.recipes.base import Recipe


@dataclass
class TrainCVResult:
    """CV 学習結果のコンテナ。"""

    oof_predictions: np.ndarray
    n_splits: int
    covered_count: int
    sample_count: int


@dataclass
class CVTrainer:
    """CV 分割を回し、OOF 予測を算出する学習器。"""

    cv: Iterable[tuple[Dataset, Dataset]]
    recipe: Recipe
    model_builder: ModelBuilder
    preprocessor_builder: PreprocessorBuilder

    def run(self, dataset: Dataset) -> TrainCVResult:
        """CV を回して OOF 予測を返す。"""
        if dataset.y is None:
            raise ValueError("TrainCV には y が必要です")

        n_samples = dataset.X.shape[0]
        oof_sum = np.zeros(n_samples, dtype=np.float64)
        oof_count = np.zeros(n_samples, dtype=np.int32)
        sample_id_to_index = {
            int(sid): idx for idx, sid in enumerate(dataset.sample_id)
        }

        n_splits = 0
        for train_ds, valid_ds in self.cv:
            preprocessor = self.preprocessor_builder.build(
                self.recipe.preprocessor_name
            )
            model = self.model_builder.build(self.recipe.model_name)

            processed_train_ds = preprocessor.fit_transform(train_ds)
            processed_valid_ds = preprocessor.transform(valid_ds)

            model.fit(processed_train_ds)
            pred = model.predict(processed_valid_ds)
            if pred.shape[0] != valid_ds.X.shape[0]:
                raise ValueError("予測件数と validation サンプル数が一致しません")

            valid_idx = np.asarray(
                [sample_id_to_index[int(sid)] for sid in valid_ds.sample_id],
                dtype=np.int64,
            )
            oof_sum[valid_idx] += pred
            oof_count[valid_idx] += 1
            n_splits += 1

        if n_splits == 0:
            raise ValueError("CV の分割結果が空です")

        oof = np.full(n_samples, np.nan, dtype=np.float32)
        covered = oof_count > 0
        oof[covered] = (oof_sum[covered] / oof_count[covered]).astype(np.float32)

        return TrainCVResult(
            oof_predictions=oof,
            n_splits=n_splits,
            covered_count=int(covered.sum()),
            sample_count=n_samples,
        )
