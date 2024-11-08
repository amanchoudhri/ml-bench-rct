import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision


class FlexibleResNet(nn.Module):
    """
    ResNet18 with flexible input size and output classes.
    Uses pretrained weights from torchvision's ResNet18 model.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (default: 3)
    """
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        
        # Start with a fresh ResNet18
        # self.model = torchvision.models.resnet18()
        self.backbone = self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity() # type: ignore
        
        # Load pretrained weights
        # pretrained = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        # Create new classification head
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before classification layer (512-dim)."""
        return self.backbone(x)
