from __future__ import annotations

import numpy as np

from src.model.eval import OOFEvaluator
from tests.helpers import make_dataset


def test_oof_evaluator_calculates_overall_and_group_rmse() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        groups=np.array([1, 1, 2, 2]),
        sample_id=np.array([10, 11, 12, 13]),
    )
    oof = np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32)
    evaluator = OOFEvaluator()

    result = evaluator.evaluate(oof, ds)

    np.testing.assert_allclose(result.overall_rmse, np.sqrt(0.5))
    np.testing.assert_allclose(result.group_rmse_by_group[1], np.sqrt(0.5))
    np.testing.assert_allclose(result.group_rmse_by_group[2], np.sqrt(0.5))
    np.testing.assert_allclose(result.group_rmse_mean, np.sqrt(0.5))
