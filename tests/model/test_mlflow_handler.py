from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from src.model.base_model import ModelType
from src.model.mlflow_handler import MLflowHandler


def test_get_flavor_returns_sklearn_module() -> None:
    flavor = MLflowHandler._get_flavor(ModelType.SKLEARN)
    assert hasattr(flavor, "log_model")
    assert hasattr(flavor, "load_model")


def test_roundtrip_sklearn_model(tmp_path: object) -> None:
    import mlflow  # type: ignore[import-untyped]

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    model = LinearRegression()
    model.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))

    with mlflow.start_run():
        uri = MLflowHandler.log_model(ModelType.SKLEARN, model, artifact_path="model")
        loaded = MLflowHandler.load_model(ModelType.SKLEARN, uri)

    pred = loaded.predict(np.array([[4]]))
    assert pred == pytest.approx([4.0], abs=0.1)


def test_pls_get_set_native_model() -> None:
    from src.data.dataset import Dataset
    from src.model.regressors.pls import PLSRegressor

    ds = Dataset(
        X=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float32),
        y=np.array([1, 2, 3, 4], dtype=np.float32),
        groups=np.array([1, 1, 2, 2]),
        sample_id=np.array([0, 1, 2, 3]),
        wavenumbers=np.array([10000, 9990, 9980], dtype=np.float32),
    )

    original = PLSRegressor(n_components=2)
    original.fit(ds)
    native = original.get_native_model()

    restored = PLSRegressor(n_components=2)
    restored.set_native_model(native)

    pred_orig = original.predict(ds)
    pred_restored = restored.predict(ds)
    np.testing.assert_array_almost_equal(pred_orig, pred_restored)
