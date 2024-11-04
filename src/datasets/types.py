"""
Data structures for dataset configuration and split management.

This module defines the core types used throughout the dataset loading system:
- Split: Enum for requesting specific dataset splits
- AvailableSplits: Enum describing what splits exist in a dataset
- DatasetConfig: Configuration class for dataset split behavior

These types work together to provide a flexible and type-safe way to configure
and request dataset splits, while abstracting away the details of how different
datasets handle their splits internally.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict

class Split(Enum):
    """Which split of the dataset to return"""
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    FULL = auto()


class AvailableSplits(Enum):
    """What splits are available in the dataset"""
    NONE = set()
    TRAIN_TEST = {Split.TRAIN, Split.TEST}
    TRAIN_VAL_TEST = {Split.TRAIN, Split.TEST}


DEFAULT_SPLIT_NAMES = {
    Split.TRAIN: "train",
    Split.VAL: "val",
    Split.TEST: "test"
}


@dataclass
class DatasetConfig:
    """
    Dataset configuration
    
    Args:
        available_splits: Enum indicating which splits are natively available
        supports_download: Whether the dataset supports downloading
        uses_train_param: Whether dataset uses train=True/False kwarg
        split_names: Dict mapping Split enum to dataset-specific split names.
                    Updates (not replaces) the default names.
        extra_params: Additional parameters to pass to dataset constructor
    """
    available_splits: AvailableSplits
    supports_download: bool = True
    uses_train_param: bool = False  # Whether dataset uses train=True/False kwarg
    split_names: Dict[Split, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Merge provided split_names with defaults after initialization"""
        # Create a new dict with defaults
        merged_names = DEFAULT_SPLIT_NAMES.copy()
        # Update with any custom names provided
        merged_names.update(self.split_names)
        # Replace the split_names with merged version
        self.split_names = merged_names

    def get_kwargs(self, split: Split) -> Dict[str, Any]:
        """Get the kwargs needed to request this split from the dataset"""
        if split == Split.FULL:
            return self.extra_params

        if self.uses_train_param:
            return {"train": split == Split.TRAIN, **self.extra_params}
            
        return {"split": self.split_names[split], **self.extra_params}

    def has_split(self, split: Split) -> bool:
        """Whether this split is natively available"""
        if split == Split.FULL:
            return self.available_splits == AvailableSplits.NONE
        return split in self.available_splits.value
