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
    """
    Configuration for handling dataset splits and their parameters.
    
    This class handles the translation between standard split types (TRAIN/TEST/VAL/FULL) 
    and dataset-specific parameters. It supports three main patterns for how datasets
    specify their splits:
    
    1. Using a 'split' parameter with names (e.g., split="train", split="test")
    2. Using a train boolean (e.g., train=True, train=False)
    3. Using custom parameters (e.g., background=True/False for Omniglot)
    
    For most datasets, you should use one of the class methods to create configurations:
        - with_split_names(): For datasets using split="train"/"test"
        - with_train_param(): For datasets using train=True/False
    
    Only use direct construction for datasets with custom split parameters.
    
    Args:
        available_splits: What splits the dataset provides natively (e.g., TRAIN_TEST)
        split_params: How to translate splits to dataset parameters. If None, defaults
            to standard split names based on available_splits.
        supports_download: Whether the dataset supports automatic download
        extra_params: Additional parameters needed for dataset construction
    
    Examples:
        >>> # Dataset with standard split names (e.g., StanfordCars):
        >>> config = DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST)
        >>> config.get_kwargs(Split.TRAIN)
        {'split': 'train'}
        
        >>> # Dataset with custom split names (e.g., Places365):
        >>> config = DatasetConfig.with_split_names(
        ...     AvailableSplits.TRAIN_TEST,
        ...     names={Split.TRAIN: "train-standard"}
        ... )
        >>> config.get_kwargs(Split.TRAIN)
        {'split': 'train-standard'}
        
        >>> # Dataset using train boolean (e.g., MNIST):
        >>> config = DatasetConfig.with_train_param()
        >>> config.get_kwargs(Split.TRAIN)
        {'train': True}
        
        >>> # Dataset with custom parameters (e.g., Omniglot):
        >>> config = DatasetConfig(
        ...     available_splits=AvailableSplits.TRAIN_TEST,
        ...     split_params={
        ...         "background": {
        ...             Split.TRAIN: True,
        ...             Split.TEST: False
        ...         }
        ...     }
        ... )
        >>> config.get_kwargs(Split.TRAIN)
        {'background': True}
        
        >>> # Dataset with no splits (e.g., EuroSAT):
        >>> config = DatasetConfig(available_splits=AvailableSplits.NONE)
        >>> config.get_kwargs(Split.FULL)
        {}
    
    Notes:
        - The FULL split is special - it never gets split parameters
        - For datasets with AvailableSplits.NONE, split_params defaults to empty
        - For other datasets, split_params defaults to standard split names
        - VAL splits will be created from TRAIN by get_dataset() if not native
    """
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
        """
        Create config for dataset using split names.
        
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
        """
        Create config for dataset using train=True/False.
        
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
