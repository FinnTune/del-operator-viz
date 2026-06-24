"""Tests for matplotlib visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualizers import (
    plot_scalar_with_gradient,
    plot_vector_with_curl,
    plot_vector_with_divergence,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def small_grid():
    x = np.linspace(-1, 1, 15)
    y = np.linspace(-1, 1, 12)
    X, Y = np.meshgrid(x, y)
    return X, Y


class TestPlotScalarWithGradient:
    def test_returns_figure(self, small_grid):
        X, Y = small_grid
        f = X**2 + Y**2
        df_dx, df_dy = 2 * X, 2 * Y
        fig = plot_scalar_with_gradient(X, Y, f, df_dx, df_dy)
        assert isinstance(fig, plt.Figure)

    def test_has_axes_and_colorbar(self, small_grid):
        X, Y = small_grid
        f = np.sin(X) * np.cos(Y)
        fig = plot_scalar_with_gradient(X, Y, f, np.cos(X), -np.sin(Y))
        assert len(fig.axes) >= 2
        assert fig.axes[0].get_title() == "Scalar field with gradient"

    def test_custom_title(self, small_grid):
        X, Y = small_grid
        f = X + Y
        title = "Custom gradient plot"
        fig = plot_scalar_with_gradient(X, Y, f, np.ones_like(X), np.ones_like(Y), title=title)
        assert fig.axes[0].get_title() == title

    def test_subsamples_arrows_on_dense_grid(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        f = X * Y
        fig = plot_scalar_with_gradient(X, Y, f, Y, X, n_arrows=10)
        quivers = [c for c in fig.axes[0].collections if c.__class__.__name__ == "Quiver"]
        assert len(quivers) == 1


class TestPlotVectorWithDivergence:
    def test_returns_figure(self, small_grid):
        X, Y = small_grid
        Fx, Fy = X, Y
        div = 2 * np.ones_like(X)
        fig = plot_vector_with_divergence(X, Y, Fx, Fy, div)
        assert isinstance(fig, plt.Figure)

    def test_symmetric_color_limits(self, small_grid):
        X, Y = small_grid
        div = X - Y
        fig = plot_vector_with_divergence(X, Y, -Y, X, div)
        contour = fig.axes[0].collections[0]
        assert contour.get_clim()[0] == pytest.approx(-contour.get_clim()[1])

    def test_custom_title(self, small_grid):
        X, Y = small_grid
        title = "Divergence view"
        fig = plot_vector_with_divergence(X, Y, X, Y, X + Y, title=title)
        assert fig.axes[0].get_title() == title


class TestPlotVectorWithCurl:
    def test_returns_figure(self, small_grid):
        X, Y = small_grid
        Fx, Fy = -Y, X
        curl = 2 * np.ones_like(X)
        fig = plot_vector_with_curl(X, Y, Fx, Fy, curl)
        assert isinstance(fig, plt.Figure)

    def test_symmetric_color_limits(self, small_grid):
        X, Y = small_grid
        curl = X**2 - Y**2
        fig = plot_vector_with_curl(X, Y, Y, X, curl)
        contour = fig.axes[0].collections[0]
        assert contour.get_clim()[0] == pytest.approx(-contour.get_clim()[1])

    def test_custom_title(self, small_grid):
        X, Y = small_grid
        title = "Curl view"
        fig = plot_vector_with_curl(X, Y, -Y, X, np.ones_like(X), title=title)
        assert fig.axes[0].get_title() == title
