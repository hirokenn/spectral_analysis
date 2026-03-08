from __future__ import annotations

import numpy as np

from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.pipeline import PreprocessingPipeline
from tests.helpers import make_dataset


def test_preprocessing_pipeline_supports_rshift_composition() -> None:
    pipeline = (
        PreprocessingPipeline() >> IdentityPreprocessor() >> IdentityPreprocessor()
    )

    assert isinstance(pipeline, PreprocessingPipeline)
    assert len(pipeline.steps) == 2


def test_identity_preprocessor_supports_rshift_composition() -> None:
    pipeline = IdentityPreprocessor() >> IdentityPreprocessor()
    ds = make_dataset(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        y=np.array([1.0], dtype=np.float32),
        sample_id=np.array([1], dtype=np.int64),
        groups=np.array([1], dtype=np.int64),
    )

    transformed = pipeline.fit_transform(ds)

    assert isinstance(pipeline, PreprocessingPipeline)
    np.testing.assert_allclose(transformed.X, ds.X)
