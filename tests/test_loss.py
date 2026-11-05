import torch
from loss import DepthLoss

def test_depth_loss_zero_on_identical_inputs():
    criterion = DepthLoss(lambda_grad=0.5)
    target = torch.rand(2, 1, 64, 64)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0))

def test_depth_loss_backward_pass():
    criterion = DepthLoss(lambda_grad=0.5)
    target = torch.rand(2, 1, 64, 64)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)

    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
