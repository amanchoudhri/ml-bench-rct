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
    # -- NO SPLITS --
    # Datasets with no splits provided out of the box
    'SEMEION': DatasetConfig(AvailableSplits.NONE),
    'SUN397': DatasetConfig(AvailableSplits.NONE),
    'Caltech101': DatasetConfig(AvailableSplits.NONE),
    'Caltech256': DatasetConfig(AvailableSplits.NONE),
    'EuroSAT': DatasetConfig(AvailableSplits.NONE),
    
    # -- TRAIN/TEST --
    # Datasets providing train/test, which use `train: bool` in the constructor
    'MNIST': DatasetConfig.with_train_param(),
    'FashionMNIST': DatasetConfig.with_train_param(),
    'KMNIST': DatasetConfig.with_train_param(),
    'QMNIST': DatasetConfig.with_train_param(),
    'USPS':DatasetConfig.with_train_param(),
    'CIFAR10': DatasetConfig.with_train_param(),
    'CIFAR100': DatasetConfig.with_train_param(),
    'EMNIST': DatasetConfig.with_train_param(
        # EMNIST has 6 different splits with various numbers of
        # observations, balanced in different ways across classes.
        # For more information, see:
        # https://www.nist.gov/itl/products-and-services/emnist-dataset
        extra_params={'split': 'bymerge'}
        ),

    # Datasets providing train/test, using standard split names
    'Food101': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST),
    'GTSRB': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST),
    'StanfordCars': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST, supports_download=False),
    'FER2013': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST, supports_download=False),
    'SVHN': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST), # really train/test/extra, but we ignore extra.
    'LFWPeople': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST), # really train/test/10fold, but ignore 10fold.
    'STL10': DatasetConfig.with_split_names(AvailableSplits.TRAIN_TEST), # really train/test/unlabeled/train+unlabeled

    # Datasets providing train/test, with custom split names
    'Imagenette': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_TEST,
        name_overrides={Split.TEST: 'val'}
        ),
    'Places365': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_TEST,
        name_overrides={
            Split.TRAIN: 'train-standard',
            Split.TEST: 'val'
            }
        ),
    'INaturalist': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_TEST,
        name_overrides={
            Split.TRAIN: '2021_train',
            Split.TEST: '2021_valid'
            }
        ),
    'OxfordIIITPet': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_TEST,
        name_overrides={
            Split.TRAIN: 'trainval',
            Split.TEST: 'test'
            }
        ),
    # Omniglot requires a custom parameter, `background`
    # to specify train or test
    "Omniglot": DatasetConfig(
        available_splits=AvailableSplits.TRAIN_TEST,
        split_params={
            "background": {
                Split.TRAIN: True,
                Split.TEST: False
            }
        }
    ),

    # -- TRAIN/VAL/TEST --
    # Datasets providing train/val/test with standard naming
    'DTD': DatasetConfig.with_split_names(AvailableSplits.TRAIN_VAL_TEST),
    'Flowers102': DatasetConfig.with_split_names(AvailableSplits.TRAIN_VAL_TEST),
    'PCAM': DatasetConfig.with_split_names(AvailableSplits.TRAIN_VAL_TEST),
    'RenderedSST2': DatasetConfig.with_split_names(AvailableSplits.TRAIN_VAL_TEST),
    'FGVCAircraft': DatasetConfig.with_split_names(AvailableSplits.TRAIN_VAL_TEST),

    # Datasets providing train/val/test, where val is called 'valid'
    'Country211': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_VAL_TEST,
        name_overrides={Split.VAL: 'valid'}
        ),
    'CelebA': DatasetConfig.with_split_names(
        AvailableSplits.TRAIN_VAL_TEST,
        name_overrides={Split.VAL: 'valid'},
        extra_params={'target_type': 'identity'}
        ),
}
