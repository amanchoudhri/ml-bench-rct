from typing import Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def interpolate_pos_embed(pos_embed: torch.Tensor, n_patches_new: Tuple[int, int], n_patches_old: Tuple[int, int] = (14, 14)) -> torch.Tensor:
    """
    Interpolate position embeddings from one size to another.
    Args:
        pos_embed: Position embedding tensor of shape (1, n_tokens, hidden_dim)
        n_patches_new: New grid size (height, width) in terms of patches
        n_patches_old: Original grid size, defaults to 14x14 (224/16 = 14)
    Returns:
        Interpolated position embedding tensor
    """
    # Handle class token separately
    cls_token = pos_embed[:, 0:1, :]
    pos_tokens = pos_embed[:, 1:, :]

    # Treat positions as 2D grid
    h_old, w_old = n_patches_old
    h_new, w_new = n_patches_new
    pos_tokens = pos_tokens.reshape(1, h_old, w_old, -1).permute(0, 3, 1, 2)

    # Interpolate over grid
    pos_tokens = F.interpolate(
        pos_tokens, size=(h_new, w_new), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_new * w_new, -1)

    # Restore class token
    pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
    return pos_embed


class FlexibleViT(nn.Module):
    """
    Vision Transformer with flexible input size and patch size.
    Uses pretrained weights from torchvision's ViT-B/16 model.
    
    Args:
        image_size: Tuple of (height, width)
        patch_size: Size of patches to extract (must divide both height and width)
        num_classes: Number of output classes
        dropout: Dropout rate (default: 0.0)
        max_sequence_length: Maximum allowed sequence length (default: 1024)
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        num_classes: int,
        dropout: float = 0.0,
        max_sequence_length: int = 1024
    ):
        super().__init__()
        
        # Validate inputs
        h, w = image_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"Patch size {patch_size} must divide both height {h} and width {w}"
            )
            
        # Load pretrained model
        pretrained = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        hidden_dim = pretrained.encoder.layers[0].ln_1.weight.shape[0]  # Should be 768
        
        # Create patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Initialize patch embedding with xavier uniform
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        
        # Calculate sequence length
        self.n_patches = (h // patch_size) * (w // patch_size)
        n_patches_new = (h // patch_size, w // patch_size)
        
        if self.n_patches > max_sequence_length:
            warnings.warn(
                f"Sequence length {self.n_patches} exceeds maximum {max_sequence_length}. "
                f"Consider using a larger patch size."
            )
        
        # Copy class token
        self.cls_token = nn.Parameter(pretrained.class_token.clone())
        
        # Interpolate position embeddings
        pos_embed = pretrained.encoder.pos_embedding
        pos_embed_new = interpolate_pos_embed(pos_embed, n_patches_new)
        self.register_buffer('pos_embed', pos_embed_new)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Create new encoder with same architecture
        self.encoder = torchvision.models.vision_transformer.Encoder(
            seq_length=self.n_patches + 1,
            num_layers=len(pretrained.encoder.layers),
            num_heads=pretrained.encoder.layers[0].self_attention.num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=pretrained.encoder.layers[0].mlp[0].out_features,
            dropout=dropout,
            attention_dropout=pretrained.encoder.dropout.p
        )
        
        # Load pretrained encoder weights
        encoder_state_dict = pretrained.encoder.state_dict()
        # Remove position embedding and cls token from state dict as we handle those separately
        encoder_state_dict = {
            k: v for k, v in encoder_state_dict.items() 
            if not any(x in k for x in ['pos_embedding', 'cls_token'])
        }
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        
        # New classification head
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create patches
        x = self.patch_embed(x)  # B, C, H, W
        
        # Reshape to sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = x.reshape(B, H * W, C)  # B, N, C
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer encoder
        x = self.encoder(x)
        
        # Get CLS token output
        x = x[:, 0]
            
        # Classification head
        x = self.head(x)
        
        return x
