"""
Machine Learning Benchmark Randomized Control Trial

A package for conducting randomized experiments comparing CNN and Vision Transformer 
architectures across diverse image classification datasets.
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Aman Choudhri"

# Core project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

DATASETS_REGISTRY = PROJECT_ROOT / "datasets.csv"
ASSIGNMENTS_FILE = PROJECT_ROOT / "assignments.csv"

# Package-level imports for convenience
from ml_bench_rct.datasets.types import Split, AvailableSplits
from ml_bench_rct.datasets.load import get_dataset

__all__ = [
    # Core types
    "Split",
    "AvailableSplits",
    
    # Main functions
    "get_dataset",
    
    # Path constants
    "PROJECT_ROOT",
    "DATASETS_REGISTRY",
    "ASSIGNMENTS_FILE",
    
    # Metadata
    "__version__",
    "__author__",
]
