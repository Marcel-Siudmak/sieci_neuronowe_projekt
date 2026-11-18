from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode, Normalize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_pair(
    image: torch.Tensor,
    depth: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Skaluje parę (image, depth) do zadanego rozmiaru (H, W)."""
    target_size = list(size)
    if tuple(image.shape[-2:]) != size:
        image = TF.resize(
            image,
            target_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    if tuple(depth.shape[-2:]) != size:
        depth = TF.resize(
            depth,
            target_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).clamp(0.0, 1.0)
    return image, depth


class EvalTransform:
    """Deterministyczna transformacja dla walidacji i testu: resize + normalizacja ImageNet."""

    def __init__(self, size: Tuple[int, int] = (240, 320)):
        self.size = (int(size[0]), int(size[1]))
        self.normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, depth = _resize_pair(image, depth, self.size)
        image = self.normalize(image)
        return image, depth


class TrainTransform:
    """Transformacja treningowa: resize, zsynchronizowany flip poziomy, ColorJitter i normalizacja RGB."""

    def __init__(
        self,
        size: Tuple[int, int] = (240, 320),
        flip_prob: float = 0.5,
        color_jitter: bool = True,
    ):
        self.size = (int(size[0]), int(size[1]))
        self.flip_prob = flip_prob
        self.color_jitter = (
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if color_jitter
            else None
        )
        self.normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, depth = _resize_pair(image, depth, self.size)

        if torch.rand(1).item() < self.flip_prob:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = self.normalize(image)
        return image, depth


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Odwraca normalizację ImageNet dla tensora (3, H, W) lub (B, 3, H, W) do zakresu [0, 1]."""
    if image.ndim == 3:
        shape = (3, 1, 1)
    elif image.ndim == 4:
        shape = (1, 3, 1, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image.ndim}D tensor.")

    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(shape)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(shape)
    return (image * std + mean).clamp(0.0, 1.0)
