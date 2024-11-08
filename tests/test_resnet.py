import pytest
import torch
import torchvision

from ml_bench_rct.models.resnet import FlexibleResNet

@pytest.fixture
def device():
    """Return available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model_configs():
    """Common model configurations for testing"""
    return [
        # (input_channels, num_classes, input_size)
        (3, 1000, (224, 224)),    # Standard ImageNet
        (1, 10, (28, 28)),        # MNIST-like
        (3, 100, (32, 32)),       # CIFAR-like
        (3, 47, (512, 512)),      # Large square
        (3, 37, (256, 384)),      # Non-square
    ]


def test_pretrained_weights():
    """Test that pretrained weights are properly loaded"""
    model = FlexibleResNet(num_classes=1000)
    pretrained = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    # Check intermediate layer weights match
    model_state = model.model.state_dict()
    pretrained_state = pretrained.state_dict()
    
    # Layer2.0.conv1 is a good layer to check - it's not the first or last layer
    layer_to_check = 'layer2.0.conv1.weight'
    assert torch.allclose(
        model_state[layer_to_check],
        pretrained_state[layer_to_check]
    ), f"Weights for {layer_to_check} don't match pretrained"


def test_modified_first_layer():
    """Test that first layer is properly modified for different input channels"""
    model = FlexibleResNet(num_classes=10, input_channels=1)
    
    # Check first conv layer has correct in_channels
    assert model.model.conv1.in_channels == 1, \
        f"Expected 1 input channel, got {model.model.conv1.in_channels}"
    
    # Check weights are initialized (not zeros)
    assert not torch.allclose(
        model.model.conv1.weight,
        torch.zeros_like(model.model.conv1.weight)
    ), "First layer weights appear to be zeros"


@pytest.mark.parametrize(
    "input_channels,num_classes,input_size",
    [
        (3, 10, (224, 224)),
        (1, 10, (28, 28)),
    ]
)
def test_model_training(input_channels, num_classes, input_size, device):
    """Test model can be trained (weights updated)"""
    model = FlexibleResNet(
        num_classes=num_classes,
        input_channels=input_channels
    ).to(device)
    
    # Save initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.clone()
    
    # Perform one training step
    batch_size = 4
    x = torch.randn(batch_size, input_channels, *input_size).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # Verify weights have been updated
    weights_updated = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_weights[name]):
            weights_updated = True
            break
    
    assert weights_updated, "No weights were updated after training step"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_independence(batch_size, device):
    """Test that samples in a batch are processed independently"""
    model = FlexibleResNet(
        num_classes=10,
        input_channels=3
    ).to(device)
    
    # Create a batch where one sample is zeros
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    x[0] = torch.zeros_like(x[0])
    
    with torch.no_grad():
        out = model(x)
    
    # Check that output for zero sample is different from others
    for i in range(1, batch_size):
        assert not torch.allclose(out[0], out[i]), \
            "Model producing same output for different inputs"
