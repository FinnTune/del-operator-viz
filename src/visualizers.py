"""
Plotting helpers for del-operator visualizations.

These wrap matplotlib calls into named, readable functions so the notebooks
can focus on the math rather than on plot configuration. Every function takes
a 2D grid and returns the matplotlib ``Figure`` so you can further customize.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_scalar_with_gradient(
    X: np.ndarray,
    Y: np.ndarray,
    f: np.ndarray,
    df_dx: np.ndarray,
    df_dy: np.ndarray,
    title: str = "Scalar field with gradient",
    n_arrows: int = 20,
    cmap: str = "viridis",
):
    """Filled contour of f with overlaid gradient quiver arrows.

    The ``n_arrows`` parameter controls how many arrows are drawn along each
    axis — drawing one arrow per grid cell looks like a hairy mess on dense
    grids, so we subsample.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    contour = ax.contourf(X, Y, f, levels=30, cmap=cmap)
    fig.colorbar(contour, ax=ax, label="f(x, y)")

    ax.contour(X, Y, f, levels=10, colors="white", linewidths=0.5, alpha=0.6)

    step_y = max(1, X.shape[0] // n_arrows)
    step_x = max(1, X.shape[1] // n_arrows)
    sl = (slice(None, None, step_y), slice(None, None, step_x))

    ax.quiver(
        X[sl], Y[sl],
        df_dx[sl], df_dy[sl],
        color="white", angles="xy", scale_units="xy",
        width=0.004, alpha=0.9,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


def plot_vector_with_divergence(
    X: np.ndarray,
    Y: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    div: np.ndarray,
    title: str = "Vector field with divergence",
    cmap: str = "RdBu_r",
):
    """Streamlines of F over a divergence heatmap (red=source, blue=sink)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Symmetric color limits so 0 lands at white
    vmax = np.max(np.abs(div))
    contour = ax.contourf(X, Y, div, levels=30, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(contour, ax=ax, label="∇·F")

    ax.streamplot(X, Y, Fx, Fy, color="black", density=1.4, linewidth=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


def plot_vector_with_curl(
    X: np.ndarray,
    Y: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    curl: np.ndarray,
    title: str = "Vector field with curl",
    cmap: str = "PiYG",
):
    """Streamlines over a 2D-curl heatmap (green=CCW, magenta=CW)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    vmax = np.max(np.abs(curl))
    contour = ax.contourf(X, Y, curl, levels=30, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(contour, ax=ax, label="(∇×F)_z")

    ax.streamplot(X, Y, Fx, Fy, color="black", density=1.4, linewidth=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig
