from __future__ import annotations

import numpy as np

from src.model.trainers.cv_trainer import CVTrainer
from tests.helpers import (
    DummyModelBuilder,
    DummyPreprocessorBuilder,
    make_dataset,
    make_recipe,
    subset,
)


def test_cv_trainer_returns_expected_oof_predictions() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        groups=np.array([1, 1, 2, 2]),
        sample_id=np.array([10, 11, 12, 13]),
    )
    cv = [
        (subset(ds, [0, 1]), subset(ds, [2, 3])),
        (subset(ds, [2, 3]), subset(ds, [0, 1])),
    ]
    trainer = CVTrainer(
        cv=cv,
        recipe=make_recipe(),
        model_builder=DummyModelBuilder(),
        preprocessor_builder=DummyPreprocessorBuilder(),
    )

    result = trainer.run(ds)

    expected = np.array([3.5, 3.5, 1.5, 1.5], dtype=np.float32)
    np.testing.assert_allclose(result.oof_predictions, expected)
    assert result.n_splits == 2
    assert result.covered_count == 4
    assert result.sample_count == 4
