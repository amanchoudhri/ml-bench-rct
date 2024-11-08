"""
Dataset loading with consistent split handling.

This package provides a consistent interface for loading and splitting datasets,
abstracting away the differences in how various datasets handle their splits.

Example:
    >>> from datasets import get_dataset, Split
    >>> train_data = get_dataset('MNIST', Split.TRAIN)
    >>> val_data = get_dataset('MNIST', Split.VAL)  # Creates validation split
    >>> test_data = get_dataset('MNIST', Split.TEST)

The main interface is through get_dataset(), which handles:
- Loading datasets with consistent split semantics
- Creating missing splits when needed (e.g., validation splits)
- Maintaining deterministic split creation
- Proper handling of various dataset split configurations
"""

from ml_bench_rct.datasets.config import DATASET_CONFIGS
from ml_bench_rct.datasets.load import get_dataset
from ml_bench_rct.datasets.types import Split, AvailableSplits, DatasetConfig

__all__ = ['Split', 'get_dataset', 'DATASET_CONFIGS']
