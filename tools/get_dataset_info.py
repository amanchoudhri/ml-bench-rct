import multiprocessing as mp
import sys

from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import VisionDataset

# Add src/ to Python path for package imports
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
sys.path.append(str(src_path))

from datasets import get_dataset, Split, DATASET_CONFIGS

ROOT = Path('~/aman/classes/snr-1/experiments/practicum/src/data').expanduser()


class SizeExtractionDataset(Dataset):
    """Wrapper dataset that extracts image sizes during __getitem__"""
    def __init__(self, dataset: Union[VisionDataset, ConcatDataset]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        item = self.dataset[idx]
        # Handle both (image, label) and (image, label, *rest) formats
        img = item[0] if isinstance(item, (tuple, list)) else item
        
        if not isinstance(img, Image.Image):
            return (0, 0)  # Skip non-image items
            
        return img.size  # Returns (width, height)

def avg_img_size(dataset: Union[VisionDataset, ConcatDataset]) -> str:
    """
    Calculate the average image size in a vision dataset.
    
    Args:
        dataset: A VisionDataset or ConcatDataset containing PIL Images
        
    Returns:
        str: Average dimensions in format "widthxheight"
        
    Note:
        - Uses parallel processing via DataLoader for efficiency
        - Assumes dataset items are either (image, label) or (image, label, *_)
        - For ConcatDataset, aggregates sizes across all constituent datasets
        - Expects PIL Images as input
    """
    # Wrap dataset to extract sizes in worker processes
    size_dataset = SizeExtractionDataset(dataset)
    
    # Configure DataLoader for parallel processing
    num_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers to avoid overhead
    loader = DataLoader(
        size_dataset,
        batch_size=64,  # Can use larger batches since we're just passing tuples
        num_workers=num_workers,
        shuffle=False  # No need to shuffle for size calculation
    )
    
    # Accumulators for running sums
    total_width = 0
    total_height = 0
    total_images = 0
    
    # Process batches in parallel
    for size_batch in loader:
        widths, heights = size_batch

        valid_mask = (widths > 0) & (heights > 0)
        valid_widths = widths[valid_mask]
        valid_heights = heights[valid_mask]

        if len(valid_widths) == 0:
            continue
            
        total_width += valid_widths.sum().item()
        total_height += valid_heights.sum().item()
        total_images += len(valid_widths)
    
    # Avoid division by zero
    if total_images == 0:
        raise ValueError("No valid images found in dataset")
        
    # Calculate averages and round to nearest integer
    avg_width = int(round(total_width / total_images))
    avg_height = int(round(total_height / total_images))
    
    return f"{avg_width}x{avg_height}"


if __name__ == "__main__":
    p = ArgumentParser(
        description="Print length and average image size for a split of a dataset."
        )

    p.add_argument('dataset', choices=DATASET_CONFIGS.keys())
    p.add_argument(
        '--split',
        default='FULL',
        choices=[s.name for s in Split]
        )

    args = p.parse_args()

    dataset = get_dataset(args.dataset, split=Split[args.split])

    length = len(dataset)
    size = avg_img_size(dataset)

    print(length)
    print(size)
