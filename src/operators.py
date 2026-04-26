"""
Numerical implementations of the three del-operator actions.

All functions use central finite differences on a regular grid (NumPy's
``np.gradient`` under the hood). The point is to be transparent rather than
fast — every line should match a textbook formula.

Convention
----------
Grids are stored with **row index = y, column index = x**, which is what
``meshgrid(x, y)`` produces by default. So:

    F[i, j]  ≡  F(x_j, y_i)

When we ask ``np.gradient(F, dy, dx)``, NumPy returns derivatives in
**axis order** — ``(∂F/∂y, ∂F/∂x)``. We unpack accordingly throughout.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gradient: scalar field f(x, y[, z])  →  vector field (∂f/∂x, ∂f/∂y[, ∂f/∂z])
# ---------------------------------------------------------------------------

def gradient_2d(f: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ∇f for a 2D scalar field.

    Parameters
    ----------
    f
        Scalar field on a regular grid, shape ``(Ny, Nx)``.
    dx, dy
        Grid spacing along the x and y axes.

    Returns
    -------
    (df_dx, df_dy)
        Each component has the same shape as ``f``.
    """
    df_dy, df_dx = np.gradient(f, dy, dx)
    return df_dx, df_dy


def gradient_3d(
    f: np.ndarray, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ∇f for a 3D scalar field with shape ``(Nz, Ny, Nx)``."""
    df_dz, df_dy, df_dx = np.gradient(f, dz, dy, dx)
    return df_dx, df_dy, df_dz


# ---------------------------------------------------------------------------
# Divergence: vector field F  →  scalar field ∇·F
# ---------------------------------------------------------------------------

def divergence_2d(
    Fx: np.ndarray, Fy: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Compute ∇·F = ∂F_x/∂x + ∂F_y/∂y for a 2D vector field."""
    dFx_dx = np.gradient(Fx, dx, axis=1)
    dFy_dy = np.gradient(Fy, dy, axis=0)
    return dFx_dx + dFy_dy


def divergence_3d(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Compute ∇·F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z for a 3D vector field."""
    dFx_dx = np.gradient(Fx, dx, axis=2)
    dFy_dy = np.gradient(Fy, dy, axis=1)
    dFz_dz = np.gradient(Fz, dz, axis=0)
    return dFx_dx + dFy_dy + dFz_dz


# ---------------------------------------------------------------------------
# Curl: vector field F  →  vector field ∇×F (or scalar in 2D)
# ---------------------------------------------------------------------------

def curl_2d(
    Fx: np.ndarray, Fy: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Compute the scalar 2D curl: (∇×F)_z = ∂F_y/∂x − ∂F_x/∂y.

    This is the only non-zero component when F lies entirely in the xy-plane.
    Positive values mean counter-clockwise rotation, negative means clockwise.
    """
    dFy_dx = np.gradient(Fy, dx, axis=1)
    dFx_dy = np.gradient(Fx, dy, axis=0)
    return dFy_dx - dFx_dy


def curl_3d(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the full 3D curl ∇×F = (curl_x, curl_y, curl_z).

    Component formulas:
        curl_x = ∂F_z/∂y − ∂F_y/∂z
        curl_y = ∂F_x/∂z − ∂F_z/∂x
        curl_z = ∂F_y/∂x − ∂F_x/∂y
    """
    dFz_dy = np.gradient(Fz, dy, axis=1)
    dFy_dz = np.gradient(Fy, dz, axis=0)
    dFx_dz = np.gradient(Fx, dz, axis=0)
    dFz_dx = np.gradient(Fz, dx, axis=2)
    dFy_dx = np.gradient(Fy, dx, axis=2)
    dFx_dy = np.gradient(Fx, dy, axis=1)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    return curl_x, curl_y, curl_z
