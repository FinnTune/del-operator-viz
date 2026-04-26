"""Del operator visualizer — numerical and visualization helpers."""

from .operators import (
    gradient_2d,
    gradient_3d,
    divergence_2d,
    divergence_3d,
    curl_2d,
    curl_3d,
)
from .visualizers import (
    plot_scalar_with_gradient,
    plot_vector_with_divergence,
    plot_vector_with_curl,
)

__all__ = [
    "gradient_2d",
    "gradient_3d",
    "divergence_2d",
    "divergence_3d",
    "curl_2d",
    "curl_3d",
    "plot_scalar_with_gradient",
    "plot_vector_with_divergence",
    "plot_vector_with_curl",
]
