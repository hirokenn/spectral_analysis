from __future__ import annotations

import numpy as np

from src.model.plot import OOFPlotter
from tests.helpers import make_dataset


class TestCorrelationPlot:
    def test_returns_figure(self) -> None:
        plotter = OOFPlotter()
        fig = plotter.correlation_plot(
            np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 2.8])
        )
        assert fig is not None
        assert len(fig.data) == 2  # scatter + y=x line


class TestResidualHistogram:
    def test_returns_figure(self) -> None:
        plotter = OOFPlotter()
        fig = plotter.residual_histogram(
            np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 2.8])
        )
        assert fig is not None
        assert len(fig.data) == 1


class TestFeatureImportancePlot:
    def test_with_feature_importance_method(self) -> None:
        class FakeModel:
            def feature_importance(self) -> dict[str, float]:
                return {"f1": 0.1, "f2": 0.2, "f3": 0.3}

        plotter = OOFPlotter()
        fig = plotter.feature_importance_plot(FakeModel())
        assert fig is not None

    def test_returns_none_for_unsupported_model(self) -> None:
        plotter = OOFPlotter()
        assert plotter.feature_importance_plot("not_a_model") is None

    def test_returns_none_when_method_returns_none(self) -> None:
        class FakeModel:
            def feature_importance(self) -> None:
                return None

        plotter = OOFPlotter()
        assert plotter.feature_importance_plot(FakeModel()) is None


class TestCreateAll:
    def test_returns_at_least_two_figures(self) -> None:
        ds = make_dataset(
            X=np.array([[0.0], [1.0], [2.0]]),
            y=np.array([1.0, 2.0, 3.0]),
            sample_id=np.array([1, 2, 3]),
        )
        oof = np.array([1.1, 2.2, 2.8], dtype=np.float32)
        plotter = OOFPlotter()
        figures = plotter.create_all(oof, ds)
        assert "plots/correlation.html" in figures
        assert "plots/residual_histogram.html" in figures

    def test_includes_feature_importance_when_model_supports(self) -> None:
        ds = make_dataset(
            X=np.array([[0.0, 1.0], [1.0, 2.0]]),
            y=np.array([1.0, 2.0]),
            sample_id=np.array([1, 2]),
        )
        oof = np.array([1.0, 2.0], dtype=np.float32)

        class FakeModel:
            def feature_importance(self) -> dict[str, float]:
                return {"f1": 0.1, "f2": 0.2}

        plotter = OOFPlotter()
        figures = plotter.create_all(oof, ds, model=FakeModel())
        assert "plots/feature_importance.html" in figures

    def test_raises_when_y_is_none(self) -> None:
        ds = make_dataset(
            X=np.array([[0.0]]),
            y=None,
            sample_id=np.array([1]),
        )
        plotter = OOFPlotter()
        try:
            plotter.create_all(np.array([1.0]), ds)
            assert False, "ValueError が送出されるはず"
        except ValueError:
            pass
