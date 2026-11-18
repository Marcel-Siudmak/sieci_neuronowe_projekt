import pytest
import torch

from src.loss import DepthLoss
from src.model import DepthUNet


@pytest.mark.parametrize(
    "backbone,batch_size,height,width",
    [
        ("resnet18", 2, 240, 320),
        ("resnet18", 1, 480, 640),
        ("resnet34", 2, 240, 320),
    ],
)
def test_depth_unet_backbones_and_resolutions(backbone: str, batch_size: int, height: int, width: int):
    model = DepthUNet(backbone=backbone, pretrained=False)
    model.eval()
    x = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 1, height, width)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


def test_depth_unet_arbitrary_resolution():
    model = DepthUNet(backbone="resnet18", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 123, 165)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1, 123, 165)
    assert torch.isfinite(out).all()
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


def test_depth_unet_backward_with_depth_loss():
    model = DepthUNet(backbone="resnet18", pretrained=False)
    model.train()
    criterion = DepthLoss(lambda_grad=1.0)

    x = torch.randn(2, 3, 120, 160)
    target = torch.rand(2, 1, 120, 160)

    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_depth_unet_invalid_backbone():
    with pytest.raises(ValueError):
        DepthUNet(backbone="unknown_net", pretrained=False)
