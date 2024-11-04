"""
Dataset-specific configurations.

This module contains the configurations for all supported datasets, specifying:
- What splits are available in each dataset
- How to access those splits
- Any custom split names or extra parameters needed

To add support for a new dataset, add it to the DATASET_CONFIGS dictionary
with appropriate configuration.
"""

from datasets.types import AvailableSplits, DatasetConfig, Split

# Dataset configurations
DATASET_CONFIGS = {
    # No split datasets
    'SEMEION': DatasetConfig(AvailableSplits.NONE),
    'SUN397': DatasetConfig(AvailableSplits.NONE),
    'Caltech101': DatasetConfig(AvailableSplits.NONE),
    'Caltech256': DatasetConfig(AvailableSplits.NONE),
    'EuroSAT': DatasetConfig(AvailableSplits.NONE),
    
    # Train/test split datasets
    'Food101': DatasetConfig(AvailableSplits.TRAIN_TEST),
    'GTSRB': DatasetConfig(AvailableSplits.TRAIN_TEST),
    'StanfordCars': DatasetConfig(AvailableSplits.TRAIN_TEST, supports_download=False),
    'FER2013': DatasetConfig(AvailableSplits.TRAIN_TEST, supports_download=False),
    'SVHN': DatasetConfig(AvailableSplits.TRAIN_TEST), # really train/test/extra, but we ignore extra.
    'LFWPeople': DatasetConfig(AvailableSplits.TRAIN_TEST), # really train/test/10fold, but ignore 10fold.
    'STL10': DatasetConfig(AvailableSplits.TRAIN_TEST), # really train/test/unlabeled/train+unlabeled
    
    # Train/test datasets that use `train: bool` kwarg in the constructor
    'MNIST': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'FashionMNIST': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'KMNIST': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'QMNIST': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'USPS': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'CIFAR10': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'CIFAR100': DatasetConfig(AvailableSplits.TRAIN_TEST, uses_train_param=True),
    'EMNIST': DatasetConfig(
        AvailableSplits.TRAIN_TEST,
        uses_train_param=True,
        extra_params={'split': 'bymerge'}
        ),

    # Train/val/test split
    'DTD': DatasetConfig(AvailableSplits.TRAIN_VAL_TEST),
    'Flowers102': DatasetConfig(AvailableSplits.TRAIN_VAL_TEST),
    'PCAM': DatasetConfig(AvailableSplits.TRAIN_VAL_TEST),
    'RenderedSST2': DatasetConfig(AvailableSplits.TRAIN_VAL_TEST),
    'FGVCAircraft': DatasetConfig(AvailableSplits.TRAIN_VAL_TEST),

    # Train/valid/test split
    'Country211': DatasetConfig(
        AvailableSplits.TRAIN_VAL_TEST,
        split_names={Split.VAL: 'valid'}
        ),
    'CelebA': DatasetConfig(
        AvailableSplits.TRAIN_VAL_TEST,
        split_names={Split.VAL: 'valid'},
        extra_params={'target_type': 'identity'}
        ),

    # CUSTOM
    'Imagenette': DatasetConfig(
        AvailableSplits.TRAIN_TEST,
        split_names={Split.TEST: 'val'}
        ),
    'Places365': DatasetConfig(
        AvailableSplits.TRAIN_TEST,
        split_names={
            Split.TRAIN: 'train-standard',
            Split.TEST: 'val'
            }
        ),
    'INaturalist': DatasetConfig(
        AvailableSplits.TRAIN_TEST,
        split_names={
            Split.TRAIN: '2021_train',
            Split.TEST: '2021_valid'
            }
        ),
    'OxfordIIITPet': DatasetConfig(
        AvailableSplits.TRAIN_TEST,
        split_names={
            Split.TRAIN: 'trainval',
            Split.TEST: 'test'
            }
        ),
}
