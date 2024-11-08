"""
Neural network architectures for image classification benchmarking.

This module provides implementations of CNN and Vision Transformer architectures
specifically designed for systematic performance comparison across diverse image 
classification tasks.

Key Features:
    - Dynamic input size handling for both architectures
    - Consistent initialization from pretrained weights
    - Flexible configuration of key architectural parameters
    - Uniform interface across model types

Available Models:
    FlexibleResNet: A ResNet18-based CNN that handles arbitrary input sizes
    and class counts while leveraging ImageNet pretrained weights.
    
    FlexibleViT: A ViT-B/16 variant that supports custom patch sizes and
    input dimensions while maintaining pretrained knowledge.

Usage:
    from ml_bench_rct.models import FlexibleResNet, FlexibleViT
    
    # Initialize CNN
    cnn = FlexibleResNet(
        num_classes=10,
        input_channels=3
    )
    
    # Initialize ViT
    vit = FlexibleViT(
        image_size=(224, 224),
        patch_size=16,
        num_classes=10,
        dropout=0.1
    )

References:
    ResNet: https://arxiv.org/abs/1512.03385
    ViT: http://arxiv.org/abs/2010.11929
"""

from ml_bench_rct.models.vit import FlexibleViT
from ml_bench_rct.models.resnet import FlexibleResNet

__all__ = ["FlexibleViT", "FlexibleResNet"]

