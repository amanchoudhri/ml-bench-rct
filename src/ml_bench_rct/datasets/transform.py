"""
Custom transforms for dataset preprocessing.

This module contains transforms that aren't available in torchvision but are
needed for our specific preprocessing needs, particularly around handling
different channel configurations.
"""

import torch
from torch import nn


class ChannelTransform(nn.Module):
    """
    Transform to ensure consistent 3-channel output.
    
    This transform handles:
    1. Single-channel → RGB by expanding
    2. RGBA → RGB by dropping alpha
    3. Passing through existing RGB
    
    Args:
        expected_channels: Number of channels in source images
        warning: If True, warn when encountering unexpected channels
        
    Example:
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     ChannelTransform(ImageChannels.SINGLE),
        ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ...                         std=[0.229, 0.224, 0.225])
        ... ])
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError(
                f"Expected tensor input, got {type(x)}. "
                "ChannelTransform must be applied after ToTensor()"
            )
            
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D tensor (C,H,W), got shape {x.shape}"
            )
            
        c = x.shape[0]
        
        # Convert to 3 channels
        if c == 1:
            # Expand grayscale to RGB
            return x.expand(3, -1, -1)
        elif c == 4:
            # Drop alpha channel
            return x[:3]
        elif c == 3:
            # Already RGB
            return x
        else:
            raise ValueError(f"Unsupported number of channels: {c}")
