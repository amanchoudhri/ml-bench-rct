#!/usr/bin/env python3

from pathlib import Path
from typing import Tuple

import pandas as pd

def compute_standardized_size(width: int, height: int, original_pixels: int) -> Tuple[int, int]:
    """
    Compute standardized dimensions with size-based patch compatibility.
    
    For smaller images (<=10k pixels), round to multiples of 4.
    For larger images (>10k pixels), round to multiples of 32.
    
    Args:
        width: Original width
        height: Original height
        original_pixels: Total pixels in original image (width * height)
        
    Returns:
        Tuple of (standardized_width, standardized_height)
    """
    # Choose patch size based on image size
    patch_size = 32 if original_pixels > 10000 else 4

    def round_to_multiple(x: int) -> int:
        return ((x + patch_size - 1) // patch_size) * patch_size
    
    return (round_to_multiple(width), round_to_multiple(height))

def main():
    # Read the CSV
    PROJECT_ROOT = Path('~/aman/classes/snr-1/experiments/practicum').expanduser()
    DATA_PATH =  PROJECT_ROOT / "datasets.csv"
    df = pd.read_csv(DATA_PATH)
    
    # Calculate standardized dimensions
    standardized_dims = df.apply(
        lambda row: pd.Series(
            compute_standardized_size(
                row['width'],
                row['height'],
                row['img_size']
                ),
            index=['standardized_width', 'standardized_height']
        ),
        axis=1
    )
    
    # Add to dataframe
    df['standardized_width'] = standardized_dims['standardized_width']
    df['standardized_height'] = standardized_dims['standardized_height']
    
    # Print changes for review
    print("\nProposed standardized dimensions:")
    for _, row in df.iterrows():
        print(f"\n{row['Dataset']}:")
        print(f"  Original: {row['width']}x{row['height']}")
        print(f"  Standardized: {row['standardized_width']}x{row['standardized_height']}")
        print(f"  Compatible with patch sizes: {[p for p in [4,8,16,32] if row['standardized_width'] % p == 0 and row['standardized_height'] % p == 0]}")
    
    # Save back to CSV
    df.to_csv("datasets.csv", index=False)
    print("\nUpdated datasets.csv with standardized dimensions.")

if __name__ == "__main__":
    main()
