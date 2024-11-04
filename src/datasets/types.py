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
from typing import Any, Dict, Optional

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
    TRAIN_VAL_TEST = {Split.TRAIN, Split.VAL, Split.TEST}


DEFAULT_SPLIT_NAMES = {
    Split.TRAIN: "train",
    Split.VAL: "val",
    Split.TEST: "test"
}


@dataclass
class DatasetConfig:
    """Configuration for how to load different splits of a dataset."""
    available_splits: AvailableSplits
    split_params: Dict[str, Dict[Split, Any]] = field(default_factory=dict)
    supports_download: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.split_params and self.available_splits != AvailableSplits.NONE:
            raise ValueError("Must specify split_params for datasets with splits")
            
        # Validate that split_params matches available_splits
        if self.split_params:
            available_params = {
                split 
                for split_values in self.split_params.values()
                for split in split_values.keys()
            }
            if available_params != self.available_splits.value:
                raise ValueError(
                    f"split_params splits {available_params} don't match "
                    f"available_splits {self.available_splits.value}"
                )

    def get_kwargs(self, split: Split) -> Dict[str, Any]:
        """Get the parameters needed to request a specific split."""
        kwargs = {}
        
        if split != Split.FULL:
            for param_name, split_values in self.split_params.items():
                if split in split_values:
                    kwargs[param_name] = split_values[split]
        
        kwargs.update(self.extra_params)
        return kwargs

    def has_split(self, split: Split) -> bool:
        """Check if this split is natively available."""
        if split == Split.FULL:
            return self.available_splits == AvailableSplits.NONE
        return split in self.available_splits.value

    @classmethod
    def with_split_names(
        cls,
        available_splits: AvailableSplits,
        name_overrides: Optional[Dict[Split, str]] = None,
        **kwargs
    ) -> 'DatasetConfig':
        """Create config for dataset using split names.
        
        Most datasets use standard names for their splits:
          - TRAIN split uses "train"
          - TEST split uses "test" 
          - VAL split uses "val" (for TRAIN_VAL_TEST datasets)
        
        This method lets you override these defaults for datasets that
        use different names (e.g., "train-standard" instead of "train").

        Args:
            available_splits: What splits are available in this dataset
            name_overrides: Optional mapping to override default split names
                          Only specify the splits that differ from defaults
            **kwargs: Additional configuration parameters

        Examples:
            # Standard split names (train/test):
            >>> DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST)

            # Override test split name only:
            >>> DatasetConfig.with_split_names(
            ...     AvailableSplits.TRAIN_TEST,
            ...     name_overrides={Split.TEST: "val"}
            ... )

            # Custom names for both splits:
            >>> DatasetConfig.with_split_names(
            ...     AvailableSplits.TRAIN_TEST,
            ...     name_overrides={
            ...         Split.TRAIN: "train-standard",
            ...         Split.TEST: "test-hard"
            ...     }
            ... )
        """
        if available_splits == AvailableSplits.NONE:
            return cls(available_splits=available_splits, **kwargs)

        # Start with default names for available splits
        split_names = {
            Split.TRAIN: "train",
            Split.TEST: "test"
        }
        if available_splits == AvailableSplits.TRAIN_VAL_TEST:
            split_names[Split.VAL] = "val"
            
        # Override with any custom names
        if name_overrides:
            split_names.update(name_overrides)
        
        return cls(
            available_splits=available_splits,
            split_params={"split": split_names},
            **kwargs
        )

    @classmethod
    def with_train_param(cls, **kwargs) -> 'DatasetConfig':
        """Create config for dataset using train=True/False.
        
        Example:
            >>> config = DatasetConfig.with_train_param(
            ...     extra_params={"split": "bymerge"}
            ... )
        """
        return cls(
            available_splits=AvailableSplits.TRAIN_TEST,
            split_params={
                "train": {
                    Split.TRAIN: True,
                    Split.TEST: False
                }
            },
            **kwargs
        )

# @dataclass
# class DatasetConfig:
#     """
#     Configuration for how to load different splits of a dataset.
#     
#     There are three ways datasets typically handle splits:
#     1. Using a 'split' parameter with names like "train"/"test"
#     2. Using a train=True/False parameter
#     3. Using custom parameters specific to that dataset
#     
#     Args:
#         available_splits: What splits the dataset natively provides
#         supports_download: Whether the dataset supports automatic download
#         uses_train_param: Dataset uses train=True/False instead of split names
#         split_names: Maps our Split enum to dataset-specific split names
#         split_params: Maps our Split enum to dataset-specific parameters
#         extra_params: Additional parameters needed for dataset construction
#     
#     Examples:
#         # Standard dataset using split names:
#         DatasetConfig(
#             available_splits=AvailableSplits.TRAIN_TEST,
#             split_names={Split.TRAIN: "train-standard", Split.TEST: "test"}
#         )
#         
#         # Dataset using train=True/False:
#         DatasetConfig(
#             available_splits=AvailableSplits.TRAIN_TEST,
#             uses_train_param=True
#         )
#         
#         # Omniglot's custom background parameter:
#         DatasetConfig(
#             available_splits=AvailableSplits.TRAIN_TEST,
#             split_params={
#                 "background": {  # Parameter name
#                     Split.TRAIN: True,   # For training set
#                     Split.TEST: False    # For test set
#                 }
#             }
#         )
#     """
#     available_splits: AvailableSplits
#     supports_download: bool = True
#     uses_train_param: bool = False  # Whether dataset uses train=True/False kwarg
#     split_names: Dict[Split, str] = field(default_factory=dict)
#     split_params: Dict[str, Dict[Split, Any]] = field(default_factory=dict)
#     extra_params: Dict[str, Any] = field(default_factory=dict)
#
#     def __post_init__(self):
#         """Merge provided split_names with defaults after initialization"""
#         # Create a new dict with defaults
#         merged_names = DEFAULT_SPLIT_NAMES.copy()
#         # Update with any custom names provided
#         merged_names.update(self.split_names)
#         # Replace the split_names with merged version
#         self.split_names = merged_names
#
#     def get_kwargs(self, split: Split) -> Dict[str, Any]:
#         """Get the kwargs needed to request this split from the dataset"""
#         if split == Split.FULL:
#             return self.extra_params
#
#         if self.uses_train_param:
#             return {"train": split == Split.TRAIN, **self.extra_params}
#             
#         return {"split": self.split_names[split], **self.extra_params}
#
#     def has_split(self, split: Split) -> bool:
#         """Whether this split is natively available"""
#         if split == Split.FULL:
#             return self.available_splits == AvailableSplits.NONE
#         return split in self.available_splits.value
