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


def _auto_arrow_length(X: np.ndarray, *components: np.ndarray, frac: float = 0.4) -> float:
    """Pick a 3D quiver arrow length so the largest arrow spans ``frac`` of the domain.

    Unlike 2D quiver (which has ``scale_units="xy"`` to size arrows in data
    coordinates automatically), ``Axes3D.quiver`` takes a single fixed
    ``length``. We derive one from the field's own magnitude so steeper
    gradients or stronger flows still show up as visibly longer arrows.
    """
    extent = X.max() - X.min()
    magnitude = np.sqrt(sum(c**2 for c in components))
    max_mag = max(np.max(magnitude), 1e-12)
    return frac * extent / max_mag


def plot_gradient_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    df_dx: np.ndarray,
    df_dy: np.ndarray,
    df_dz: np.ndarray,
    f: np.ndarray | None = None,
    title: str = "3D scalar field with gradient",
    n_arrows: int = 6,
    cmap: str = "viridis",
):
    """3D quiver of ∇f, optionally with grid points colored by f.

    ``n_arrows`` subsamples along each of the three axes (a dense 3D grid
    drawn one arrow per cell is unreadable), mirroring ``n_arrows`` in
    ``plot_scalar_with_gradient``.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    step = tuple(max(1, s // n_arrows) for s in X.shape)
    sl = tuple(slice(None, None, st) for st in step)

    if f is not None:
        sc = ax.scatter(X[sl], Y[sl], Z[sl], c=f[sl], cmap=cmap, alpha=0.4, s=15)
        fig.colorbar(sc, ax=ax, label="f", shrink=0.6, pad=0.1)

    length = _auto_arrow_length(X, df_dx, df_dy, df_dz)
    ax.quiver(
        X[sl], Y[sl], Z[sl],
        df_dx[sl], df_dy[sl], df_dz[sl],
        length=length, normalize=False, color="crimson",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return fig


def plot_divergence_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    div: np.ndarray,
    title: str = "3D vector field with divergence",
    n_arrows: int = 6,
    cmap: str = "RdBu_r",
):
    """3D quiver of F with points colored by divergence (red=source, blue=sink).

    3D streamlines don't have a stable matplotlib equivalent of
    ``ax.streamplot``, so — unlike the 2D divergence/curl plots — the scalar
    field is shown as colored scatter points rather than a background fill.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    step = tuple(max(1, s // n_arrows) for s in X.shape)
    sl = tuple(slice(None, None, st) for st in step)

    vmax = np.max(np.abs(div))
    sc = ax.scatter(
        X[sl], Y[sl], Z[sl], c=div[sl], cmap=cmap, vmin=-vmax, vmax=vmax, s=25,
    )
    fig.colorbar(sc, ax=ax, label="∇·F", shrink=0.6, pad=0.1)

    length = _auto_arrow_length(X, Fx, Fy, Fz)
    ax.quiver(
        X[sl], Y[sl], Z[sl], Fx[sl], Fy[sl], Fz[sl],
        length=length, normalize=False, color="black", alpha=0.6,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return fig


def plot_curl_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    curl_x: np.ndarray,
    curl_y: np.ndarray,
    curl_z: np.ndarray,
    title: str = "3D vector field with curl",
    n_arrows: int = 6,
    flow_color: str = "black",
    curl_color: str = "crimson",
):
    """3D quiver of F (flow) overlaid with ∇×F vectors (local rotation axis).

    Each curl arrow points along the axis a tiny paddlewheel dropped at that
    point would spin around; its length is proportional to twice the local
    angular velocity, per the usual curl interpretation.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    step = tuple(max(1, s // n_arrows) for s in X.shape)
    sl = tuple(slice(None, None, st) for st in step)

    flow_length = _auto_arrow_length(X, Fx, Fy, Fz)
    ax.quiver(
        X[sl], Y[sl], Z[sl], Fx[sl], Fy[sl], Fz[sl],
        length=flow_length, normalize=False, color=flow_color, alpha=0.4, label="F",
    )

    curl_length = _auto_arrow_length(X, curl_x, curl_y, curl_z)
    ax.quiver(
        X[sl], Y[sl], Z[sl], curl_x[sl], curl_y[sl], curl_z[sl],
        length=curl_length, normalize=False, color=curl_color, label="∇×F",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left")
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
