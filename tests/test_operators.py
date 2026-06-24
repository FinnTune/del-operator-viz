"""Tests for numerical del-operator implementations."""

from __future__ import annotations

import numpy as np
import pytest

from src.operators import (
    curl_2d,
    curl_3d,
    divergence_2d,
    divergence_3d,
    gradient_2d,
    gradient_3d,
)


def _interior_slice(ndim: int) -> tuple[slice, ...]:
    """Exclude one-cell border where finite differences are one-sided."""
    inner = slice(2, -2)
    return (inner,) * ndim


class TestGradient2D:
    def test_polynomial_field_matches_analytical(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = X**2 * Y
        df_dx, df_dy = gradient_2d(f, dx, dy)

        sl = _interior_slice(2)
        np.testing.assert_allclose(df_dx[sl], 2 * X[sl] * Y[sl], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(df_dy[sl], X[sl] ** 2, rtol=1e-2, atol=1e-2)

    def test_output_shapes_match_input(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = np.sin(X) * np.cos(Y)
        df_dx, df_dy = gradient_2d(f, dx, dy)
        assert df_dx.shape == f.shape
        assert df_dy.shape == f.shape

    def test_constant_field_is_zero(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = np.full_like(X, 7.0)
        df_dx, df_dy = gradient_2d(f, dx, dy)
        np.testing.assert_allclose(df_dx, 0.0, atol=1e-10)
        np.testing.assert_allclose(df_dy, 0.0, atol=1e-10)

    def test_linear_field(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = 3 * X - 2 * Y + 5
        df_dx, df_dy = gradient_2d(f, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(df_dx[sl], 3.0, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(df_dy[sl], -2.0, rtol=1e-2, atol=1e-2)


class TestGradient3D:
    def test_linear_field(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        f = X + 2 * Y + 3 * Z
        df_dx, df_dy, df_dz = gradient_3d(f, dx, dy, dz)
        sl = _interior_slice(3)
        np.testing.assert_allclose(df_dx[sl], 1.0, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(df_dy[sl], 2.0, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(df_dz[sl], 3.0, rtol=1e-2, atol=1e-2)

    def test_output_shapes_match_input(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        f = X * Y * Z
        df_dx, df_dy, df_dz = gradient_3d(f, dx, dy, dz)
        assert df_dx.shape == f.shape
        assert df_dy.shape == f.shape
        assert df_dz.shape == f.shape


class TestDivergence2D:
    def test_polynomial_field_matches_analytical(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = X * Y
        Fy = Y**2 - X
        div = divergence_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(div[sl], 3 * Y[sl], rtol=1e-2, atol=1e-2)

    def test_rotation_field_is_incompressible(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = -Y
        Fy = X
        div = divergence_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(div[sl], 0.0, atol=1e-2)

    def test_radial_source_field(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = X
        Fy = Y
        div = divergence_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(div[sl], 2.0, rtol=1e-2, atol=1e-2)

    def test_output_shape_matches_input(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = np.sin(X)
        Fy = np.cos(Y)
        div = divergence_2d(Fx, Fy, dx, dy)
        assert div.shape == X.shape


class TestDivergence3D:
    def test_linear_field(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        Fx, Fy, Fz = X, Y, Z
        div = divergence_3d(Fx, Fy, Fz, dx, dy, dz)
        sl = _interior_slice(3)
        np.testing.assert_allclose(div[sl], 3.0, rtol=1e-2, atol=1e-2)

    def test_output_shape_matches_input(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        div = divergence_3d(X, Y, Z, dx, dy, dz)
        assert div.shape == X.shape


class TestCurl2D:
    def test_rotation_field_matches_analytical(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = -Y
        Fy = X
        curl = curl_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(curl[sl], 2.0, rtol=1e-2, atol=1e-2)

    def test_irrotational_field_is_zero(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = X
        Fy = Y
        curl = curl_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(curl[sl], 0.0, atol=1e-2)

    def test_shear_field(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Fx = Y
        Fy = np.zeros_like(Y)
        curl = curl_2d(Fx, Fy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(curl[sl], -1.0, rtol=1e-2, atol=1e-2)

    def test_output_shape_matches_input(self, grid_2d):
        X, Y, dx, dy = grid_2d
        curl = curl_2d(X, Y, dx, dy)
        assert curl.shape == X.shape


class TestCurl3D:
    def test_rotation_about_z(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        Fx = -Y
        Fy = X
        Fz = np.zeros_like(X)
        curl_x, curl_y, curl_z = curl_3d(Fx, Fy, Fz, dx, dy, dz)
        sl = _interior_slice(3)
        np.testing.assert_allclose(curl_x[sl], 0.0, atol=1e-2)
        np.testing.assert_allclose(curl_y[sl], 0.0, atol=1e-2)
        np.testing.assert_allclose(curl_z[sl], 2.0, rtol=1e-2, atol=1e-2)

    def test_output_shapes_match_input(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        curl_x, curl_y, curl_z = curl_3d(X, Y, Z, dx, dy, dz)
        assert curl_x.shape == X.shape
        assert curl_y.shape == X.shape
        assert curl_z.shape == X.shape


class TestVectorCalculusIdentities:
    def test_curl_of_gradient_is_zero(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = X**3 + Y**2 * X
        df_dx, df_dy = gradient_2d(f, dx, dy)
        curl = curl_2d(df_dx, df_dy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(curl[sl], 0.0, atol=5e-2)

    def test_divergence_of_gradient_is_laplacian(self, grid_2d):
        X, Y, dx, dy = grid_2d
        f = X**2 + Y**2
        df_dx, df_dy = gradient_2d(f, dx, dy)
        laplacian = divergence_2d(df_dx, df_dy, dx, dy)
        sl = _interior_slice(2)
        np.testing.assert_allclose(laplacian[sl], 4.0, rtol=1e-2, atol=1e-2)

    def test_divergence_of_curl_3d_is_zero(self, grid_3d):
        X, Y, Z, dx, dy, dz = grid_3d
        Fx = Y * Z
        Fy = X * Z
        Fz = X * Y
        curl_x, curl_y, curl_z = curl_3d(Fx, Fy, Fz, dx, dy, dz)
        div = divergence_3d(curl_x, curl_y, curl_z, dx, dy, dz)
        sl = _interior_slice(3)
        np.testing.assert_allclose(div[sl], 0.0, atol=5e-2)


class TestPackageExports:
    def test_public_api_importable(self):
        import src

        expected = {
            "gradient_2d",
            "gradient_3d",
            "divergence_2d",
            "divergence_3d",
            "curl_2d",
            "curl_3d",
            "plot_scalar_with_gradient",
            "plot_vector_with_divergence",
            "plot_vector_with_curl",
        }
        assert set(src.__all__) == expected
        for name in expected:
            assert hasattr(src, name)
