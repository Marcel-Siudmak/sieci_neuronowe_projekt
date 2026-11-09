import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):

    def __init__(self, lambda_grad: float = 1.0):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_pixel = F.l1_loss(prediction, target)
        loss_grad = self.gradient_loss(prediction, target)
        return loss_pixel + self.lambda_grad * loss_grad

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        return F.l1_loss(dy_pred, dy_target) + F.l1_loss(dx_pred, dx_target)
