"""Shared fixtures for del-operator-viz tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def grid_2d():
    """Regular 2D grid with meshgrid convention (row=y, col=x)."""
    nx, ny = 41, 31
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return X, Y, dx, dy


@pytest.fixture
def grid_3d():
    """Regular 3D grid with shape (Nz, Ny, Nx)."""
    nx, ny, nz = 21, 17, 13
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # meshgrid with indexing=ij gives (Nx, Ny, Nz); transpose to (Nz, Ny, Nx)
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    Z = np.transpose(Z, (2, 1, 0))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    return X, Y, Z, dx, dy, dz
