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

from torch import nn
from torch.utils.data import ConcatDataset, Subset, random_split

from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode

from ml_bench_rct import PROJECT_ROOT

from ml_bench_rct.datasets.types import AvailableSplits, Split, DatasetConfig
from ml_bench_rct.datasets.config import DATASET_CONFIGS
from ml_bench_rct.datasets.transform import ChannelTransform

# Default data directory is under project root
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Load dataset information once at module level
DATASETS_INFO = pd.read_csv(PROJECT_ROOT / "datasets.csv")
DATASETS_INFO.set_index('dataset_name', inplace=True)


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

# def get_transform_for_dataset(dataset_name: str, 
#                             additional_transform: Optional[Callable] = None) -> transforms.Compose:
#     """
#     Create a transform pipeline for a dataset including standardized resizing.
#     """
#     info = DATASETS_INFO.loc[dataset_name]
#     target_size = (int(info['standard_width_px']), int(info['standard_height_px']))
#     
#     transform_list = [
#         transforms.Resize(target_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Can be made dataset-specific if needed
#     ]
#     
#     if additional_transform is not None:
#         transform_list.append(additional_transform)
#         print(transform_list)
#     
#     return transforms.Compose(transform_list)

def get_transform_for_dataset(
    config: DatasetConfig,
    target_size: Optional[Tuple[int, int]] = None,
    transform: Optional[nn.Module] = None,
) -> T.Compose:
    """
    Get transform pipeline for a specific dataset.

    This creates a transform pipeline that:
    1. Resizes images to target size if specified
    2. Applies any user-provided transforms
    3. Converts to tensor and handles channel normalization
    4. Applies dataset-specific normalization
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        split: Which split we're transforming
        target_size: Optional (height, width) to resize to
        transform: Optional user-provided transforms to include
        
    Returns:
        Composed transform pipeline
    
    Example:
        >>> # Basic usage
        >>> transform = get_transform_for_dataset(
        ...     'MNIST',
        ...     config,
        ...     Split.TRAIN,
        ...     target_size=(224, 224)
        ... )
        
        >>> # With custom transforms
        >>> custom_transform = transforms.Compose([
        ...     transforms.RandomHorizontalFlip(),
        ...     transforms.ColorJitter(0.2, 0.2)
        ... ])
        >>> transform = get_transform_for_dataset(
        ...     'MNIST',
        ...     config,
        ...     Split.TRAIN,
        ...     transform=custom_transform
        ... )
    """
    transforms: list[Union[Callable, nn.Module, ChannelTransform]] = []
    
    # 1. Resize if needed
    if target_size is not None:
        transforms.append(T.Resize(target_size, interpolation=InterpolationMode.BICUBIC))

    # 2. Add user transforms if provided (before tensor conversion)
    if transform is not None:
        if isinstance(transform, (list, tuple)):
            transforms.extend(transform)
        elif isinstance(transform, T.Compose):
            transforms.extend(transform.transforms)
        else:
            transforms.append(transform)

    # 3. Convert to tensor (necessary for channel handling)
    transforms.append(T.ToTensor())
    
    # 4. Handle channel count standardizing
    transforms.append(ChannelTransform())

    # 5. Normalization
    # Use the imagenet normalization
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ))

    return T.Compose(transforms)


def get_dataset(
    dataset_name: str,
    split: Split = Split.FULL,
    root: Union[str, Path] = DEFAULT_DATA_DIR,
    transform: Optional[nn.Module] = None,
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

    info = DATASETS_INFO.loc[dataset_name]
    target_size = (int(info['standard_width_px']), int(info['standard_height_px']))

    # Common parameters supported across all dataset instantiations
    common_params = {
        "root": root,
        "transform": get_transform_for_dataset(config, target_size, transform),
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
