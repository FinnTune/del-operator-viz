"""Del operator visualizer — numerical and visualization helpers."""

from .operators import (
    curl_2d,
    curl_3d,
    divergence_2d,
    divergence_3d,
    gradient_2d,
    gradient_3d,
    laplacian_2d,
    laplacian_3d,
)
from .visualizers import (
    plot_scalar_with_gradient,
    plot_vector_with_curl,
    plot_vector_with_divergence,
)

__all__ = [
    "curl_2d",
    "curl_3d",
    "divergence_2d",
    "divergence_3d",
    "gradient_2d",
    "gradient_3d",
    "laplacian_2d",
    "laplacian_3d",
    "plot_scalar_with_gradient",
    "plot_vector_with_curl",
    "plot_vector_with_divergence",
]
