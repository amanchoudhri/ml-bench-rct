import pytest

import torch
import torch.nn as nn
import torchvision.transforms as T

from ml_bench_rct import get_dataset, Split
from ml_bench_rct.models.vit import FlexibleViT, interpolate_pos_embed

@pytest.fixture
def device():
    """Return available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def model_configs():
    """Common model configurations for testing"""
    return [
        # (image_size, patch_size, num_classes, expected_seq_len)
        ((224, 224), 16, 1000, 197),  # Standard ImageNet
        ((32, 32), 4, 10, 65),        # Small (CIFAR-like)
        ((128, 256), 16, 100, 129),   # Non-square
        ((48, 48), 8, 10, 37),        # Custom
    ]

def test_invalid_patch_size():
    """Test that invalid patch sizes raise ValueError"""
    with pytest.raises(ValueError):
        FlexibleViT(
            image_size=(65, 65),  # Not divisible by 16
            patch_size=16,
            num_classes=10
        )

def test_interpolate_pos_embed():
    """Test position embedding interpolation"""
    hidden_dim = 768
    old_grid = (14, 14)
    new_grid = (7, 7)
    
    # Create dummy position embeddings
    pos_embed = torch.randn(1, 1 + old_grid[0] * old_grid[1], hidden_dim)
    
    # Interpolate
    new_pos_embed = interpolate_pos_embed(pos_embed, new_grid, old_grid)
    
    # Check shapes
    expected_length = 1 + new_grid[0] * new_grid[1]  # CLS token + grid positions
    assert new_pos_embed.shape == (1, expected_length, hidden_dim)
    
    # Check CLS token is unchanged
    assert torch.allclose(pos_embed[:, 0], new_pos_embed[:, 0])

@pytest.mark.parametrize(
    "image_size,patch_size,num_classes,expected_seq_len",
    [
        ((224, 224), 16, 1000, 197),
        ((32, 32), 4, 10, 65),
        ((128, 256), 16, 100, 129),
        ((48, 48), 8, 10, 37),
    ]
)
def test_model_shapes(image_size, patch_size, num_classes, expected_seq_len, device):
    """Test model produces correct output shapes for various configurations"""
    model = FlexibleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes
    ).to(device)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, *image_size).to(device)
    
    # Test forward pass
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, num_classes)
    
    # Test patch embedding shape
    patches = model.patch_embed(x)
    h, w = image_size[0] // patch_size, image_size[1] // patch_size
    assert patches.shape == (batch_size, 768, h, w)
    
    # Test sequence length with CLS token
    B, C, H, W = patches.shape
    patches = patches.permute(0, 2, 3, 1).reshape(B, H * W, C)
    cls_tokens = model.cls_token.expand(B, -1, -1)
    seq = torch.cat((cls_tokens, patches), dim=1)
    assert seq.shape == (batch_size, expected_seq_len, 768)

def test_pretrained_components():
    """Test that pretrained components are properly initialized"""
    model = FlexibleViT(
        image_size=(224, 224),
        patch_size=16,
        num_classes=1000
    )
    
    # Test CLS token initialization
    assert not torch.allclose(model.cls_token, torch.zeros_like(model.cls_token))
    
    # Test position embeddings
    assert hasattr(model, 'pos_embed')
    assert not torch.allclose(model.pos_embed, torch.zeros_like(model.pos_embed))
    
    # Test encoder structure
    assert len(model.encoder.layers) == 12  # Standard ViT-B/16 depth
    assert model.encoder.layers[0].self_attention.num_heads == 12  # Standard number of heads
    
    # Test hidden dimension
    assert model.head.in_features == 768  # Standard ViT-B/16 hidden dim

@pytest.mark.parametrize(
    "image_size,patch_size,num_classes",
    [
        ((224, 224), 16, 1000),
        ((32, 32), 4, 10),
    ]
)
def test_model_training(image_size, patch_size, num_classes, device):
    """Test model can be trained (weights updated)"""
    model = FlexibleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes
    ).to(device)
    
    # Save initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.clone()
    
    # Perform training step
    batch_size = 4
    x = torch.randn(batch_size, 3, *image_size).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # Verify weights were updated
    weights_updated = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_weights[name]):
            weights_updated = True
            break
    assert weights_updated, "No weights were updated after training step"

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_independence(batch_size, device):
    """Test that samples in a batch are processed independently"""
    model = FlexibleViT(
        image_size=(64, 64),
        patch_size=8,
        num_classes=10
    ).to(device)
    
    # Create batch with first sample as zeros
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    x[0] = torch.zeros_like(x[0])
    
    with torch.no_grad():
        out = model(x)
    
    # Verify different inputs produce different outputs
    for i in range(1, batch_size):
        assert not torch.allclose(out[0], out[i])

def test_dropout_effect():
    """Test that dropout has an effect during training"""
    model = FlexibleViT(
        image_size=(32, 32),
        patch_size=4,
        num_classes=10,
        dropout=0.5
    )
    
    x = torch.randn(4, 3, 32, 32)
    
    # Test outputs are different in training mode
    model.train()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert not torch.allclose(out1, out2)
    
    # Test outputs are identical in eval mode
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)

def test_large_sequence_warning():
    """Test warning is raised for large sequence lengths"""
    with pytest.warns(UserWarning, match="Sequence length .* exceeds maximum"):
        FlexibleViT(
            image_size=(512, 512),
            patch_size=8,  # Results in 4096 patches
            num_classes=10,
            max_sequence_length=1024
        )

@pytest.mark.parametrize(
    "dataset_name,image_size,patch_size,num_classes",
    [
        ("MNIST", (28, 28), 4, 10),  # Small images
        ("Imagenette", (224, 224), 16, 10),  # Standard ImageNet size
    ]
)
def test_model_forward(dataset_name, image_size, patch_size, num_classes, device):
    """Test that model can process images from different datasets."""
    # Load dataset
    try:
        dataset = get_dataset(
            dataset_name,
            Split.TRAIN,
            transform=T.Grayscale(3) if dataset_name == "MNIST" else None,
        )
    except FileNotFoundError as e:
        pytest.skip(f"Dataset {dataset_name} not found: {e}")
    
    # Create model
    model = FlexibleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes
    ).to(device)
    model.eval()
    
    # Get a small batch of data
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(images)
    
    # Verify output shape and values
    assert outputs.shape == (batch_size, num_classes), \
        f"Expected output shape {(batch_size, num_classes)}, got {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Output contains NaN values"
    assert not torch.isinf(outputs).any(), "Output contains infinity values"
    
    # Test that different images give different outputs
    output_diffs = []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            diff = torch.abs(outputs[i] - outputs[j]).mean()
            output_diffs.append(diff.item())
    
    # Verify that at least some outputs are different
    assert any(diff > 1e-6 for diff in output_diffs), \
        "All images produced identical outputs"
    
    # Test that model can process different batch sizes
    for test_batch_size in [1, 2, 8]:
        test_images = images[:test_batch_size]
        with torch.no_grad():
            test_outputs = model(test_images)
        assert test_outputs.shape == (test_batch_size, num_classes), \
            f"Failed with batch size {test_batch_size}"
