from __future__ import annotations

import pytest

from src.model.model_builder import ModelBuilder
from src.model.regressors.pls import PLSRegressor


def test_build_returns_pls_with_params_from_config() -> None:
    builder = ModelBuilder(
        config={
            "base_pls": {
                "build_type": "pls",
                "params": {"n_components": 5, "scale": False},
            }
        }
    )

    model = builder.build("base_pls")

    assert isinstance(model, PLSRegressor)
    assert model.params["n_components"] == 5
    assert model.params["scale"] is False


def test_build_single_model_returns_pls_for_base_pls() -> None:
    builder = ModelBuilder(
        config={
            "base_pls": {
                "build_type": "pls",
                "params": {"n_components": 8, "scale": True},
            }
        }
    )

    model = builder._build_single_model("base_pls")

    assert isinstance(model, PLSRegressor)
    assert model.params["n_components"] == 8


def test_build_raises_when_model_name_is_missing() -> None:
    builder = ModelBuilder(
        config={"base_pls": {"build_type": "pls", "params": {"n_components": 8}}}
    )

    with pytest.raises(ValueError, match="モデル定義が見つかりません"):
        builder.build("unknown_model")
