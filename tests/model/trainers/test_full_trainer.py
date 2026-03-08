from __future__ import annotations

import numpy as np

from src.model.trainers.full_trainer import FullTrainer
from tests.helpers import DummyModelBuilder, make_dataset, make_recipe


def test_full_trainer_fits_model_and_preprocessor() -> None:
    ds = make_dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1.0, 2.0, 3.0]),
        groups=np.array([1, 1, 2]),
        sample_id=np.array([1, 2, 3]),
    )
    trainer = FullTrainer(recipe=make_recipe(), model_builder=DummyModelBuilder())

    result = trainer.run(ds)

    transformed = result.preprocessor.transform(ds)
    pred = result.model.predict(transformed)
    np.testing.assert_allclose(pred, np.array([2.0, 2.0, 2.0], dtype=np.float32))
