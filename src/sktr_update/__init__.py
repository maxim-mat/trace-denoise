"""
SKTR: Stochastic vs Argmax Trace Recovery package.
"""

__version__ = "0.1.0"

from .core import compare_stochastic_vs_argmax_random_indices
from .run  import run_multiple_iterations

__all__ = [
    "compare_stochastic_vs_argmax_random_indices",
    "run_multiple_iterations",
]
