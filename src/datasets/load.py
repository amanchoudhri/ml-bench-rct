"""
Dataset loading functionality with consistent split handling.

This module provides the main dataset loading interface through get_dataset().
It handles:
- Loading datasets with consistent split semantics
- Creating missing splits when needed
- Maintaining deterministic split creation
- Proper handling of all dataset configuration types
"""

import hashlib

from pathlib import Path
from typing import Callable, Optional, Tuple, Type, Union

import pandas as pd
import torch
import torchvision

from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset

from datasets.config import DATASET_CONFIGS
from datasets.types import AvailableSplits, Split


# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Default data directory is under project root
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Load dataset information once at module level
DATASETS_INFO = pd.read_csv(PROJECT_ROOT / "datasets.csv")
DATASETS_INFO.set_index('Dataset', inplace=True)


def _get_split_seed(dataset_name: str, split: Split, base_seed: int) -> int:
    """
    Generate a deterministic seed for a specific split.
    
    Args:
        dataset_name: Name of the dataset
        split: Which split we're creating
        base_seed: Base random seed
        
    Returns:
        Deterministic seed for this specific split
    """
    hasher = hashlib.sha256()
    hasher.update(f"{dataset_name}_{split.name}_{base_seed}".encode())
    return int(hasher.hexdigest(), 16) % (2**32)


def create_split(
    dataset: ConcatDataset | VisionDataset | Subset[ConcatDataset | VisionDataset], 
    ratio: float, 
    dataset_name: str,
    split: Split,
    seed: int
) -> Tuple[Subset[VisionDataset | ConcatDataset], Subset[VisionDataset | ConcatDataset]]:
    """
    Create a split deterministically.

    This function splits a dataset into two parts with a deterministic split
    based on the dataset name, split type, and seed.
    
    Args:
        dataset: Dataset to split
        ratio: Fraction of data to include in the split
        dataset_name: Name of dataset (for deterministic splitting)
        split: Which split we're creating (for deterministic splitting)
        seed: Base random seed
        
    Returns:
        Tuple of (remaining_data, split_data)
    """
    n = len(dataset)
    split_size = int(n * ratio)
    remain_size = n - split_size
    
    generator = torch.Generator().manual_seed(_get_split_seed(dataset_name, split, seed))
    remain_data, split_data = random_split(
        dataset, [remain_size, split_size], generator=generator
    )
    return remain_data, split_data

def get_transform_for_dataset(dataset_name: str, 
                            additional_transform: Optional[Callable] = None) -> transforms.Compose:
    """
    Create a transform pipeline for a dataset including standardized resizing.
    """
    info = DATASETS_INFO.loc[dataset_name]
    target_size = (int(info['standardized_width']), int(info['standardized_height']))
    
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Can be made dataset-specific if needed
    ]
    
    if additional_transform is not None:
        transform_list.append(additional_transform)
    
    return transforms.Compose(transform_list)

def get_dataset(
    dataset_name: str,
    split: Split = Split.FULL,
    root: Union[str, Path] = DEFAULT_DATA_DIR,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> VisionDataset | ConcatDataset:
    """
    Load a dataset with consistent split handling.

    This function provides a consistent interface for loading dataset splits,
    regardless of how the underlying dataset handles splits. It will create
    missing splits as needed (e.g., creating a validation split from training
    data) and ensures splits are created deterministically.
    
    Args:
        dataset_name: Name of the dataset
        split: Which split to return (default: FULL)
        root: Root directory for dataset storage
        transform: Optional transform for input data
        target_transform: Optional transform for targets
        download: If True, download the dataset if not present
        val_ratio: Ratio to use for validation split if needed
        test_ratio: Ratio to use for test split if needed
        seed: Random seed for consistent splitting
    
    Returns:
        The requested dataset split
    """
    config = DATASET_CONFIGS[dataset_name]
    dataset_cls: Type[VisionDataset] = getattr(torchvision.datasets, dataset_name)


    # Common parameters supported across all dataset instantiations
    common_params = {
        "root": root,
        "transform": get_transform_for_dataset(dataset_name, transform),
        "target_transform": target_transform,
    }

    if config.supports_download:
        # Hotfix for known issue in torchvision where the download = True flag
        # raises a RuntimeError if Imagenette is already downloaded
        # https://github.com/pytorch/vision/pull/8681
        if dataset_name == "Imagenette":
            download = False
        common_params["download"] = download

    def get_native_split(split: Split) -> VisionDataset:
        """Get a natively available split from the dataset"""
        if not config.has_split(split):
            raise ValueError(f"Split {split} not available in {dataset_name}")
        return dataset_cls(**common_params, **config.get_kwargs(split))

    def get_full_dataset() -> ConcatDataset | VisionDataset:
        """Get the full dataset, either directly or by concatenating splits"""
        if config.available_splits == AvailableSplits.NONE:
            return dataset_cls(**common_params, **config.get_kwargs(Split.FULL))
        
        # Concatenate all available splits
        splits_to_concat = []
        for available_split in config.available_splits.value:
            splits_to_concat.append(get_native_split(available_split))
        return ConcatDataset(splits_to_concat)

    # Handle FULL split request
    if split == Split.FULL:
        return get_full_dataset()

    # Return native split if available
    if config.has_split(split):
        return get_native_split(split)

    # Need to create the split
    if split == Split.TEST:
        # Create test split from full data or training data
        base_data = (
            get_native_split(Split.TRAIN)
            if config.has_split(Split.TRAIN)
            else get_full_dataset()
        )
        _, test_data = create_split(base_data, test_ratio, dataset_name, Split.TEST, seed)
        return test_data

    if split == Split.VAL:
        # Create validation split from training data (after removing test if needed)
        if config.has_split(Split.TRAIN):
            train_data = get_native_split(Split.TRAIN)
        else:
            # Get full data and remove test split first
            full_data = get_full_dataset()
            train_data, _ = create_split(full_data, test_ratio, dataset_name, Split.TEST, seed)
            
        _, val_data = create_split(train_data, val_ratio, dataset_name, Split.VAL, seed)
        return val_data

    if split == Split.TRAIN:
        # Get appropriate base data
        if config.has_split(Split.TRAIN):
            base_data = get_native_split(Split.TRAIN)
        else:
            # Remove test split first if needed
            full_data = get_full_dataset()
            base_data, _ = create_split(full_data, test_ratio, dataset_name, Split.TEST, seed)
        
        # Remove validation split and return the remaining training data
        train_data, _ = create_split(base_data, val_ratio, dataset_name, Split.VAL, seed)
        return train_data
