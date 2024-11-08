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
        input_channels: int = 3,
    ):
        super().__init__()
        
        # Start with a fresh ResNet18
        self.model = torchvision.models.resnet18()
        
        # Load pretrained weights
        pretrained = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        # If input channels different from 3, create new first conv layer
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            # Initialize with Kaiming initialization
            nn.init.kaiming_normal_(
                self.model.conv1.weight,
                mode='fan_out',
                nonlinearity='relu'
            )
            
            # Load all weights except first conv and final FC
            pretrained_dict = pretrained.state_dict()
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and 'conv1' not in k and 'fc' not in k
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            # Load all weights except final FC
            pretrained_dict = pretrained.state_dict()
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and 'fc' not in k
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        # Replace final linear layer
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
