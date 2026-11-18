import csv
from pathlib import Path

import pytest
import torch
import torchvision.transforms.functional as TF
from torchvision.io import write_png
from torchvision.transforms import InterpolationMode, Normalize

from src.data_loader import get_dataloaders
from src.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EvalTransform,
    TrainTransform,
    denormalize_image,
)


def test_eval_transform_resizes_and_normalizes():
    image = torch.rand(3, 480, 640, dtype=torch.float32)
    depth = torch.rand(1, 480, 640, dtype=torch.float32)

    transform = EvalTransform(size=(240, 320))
    out_img, out_depth = transform(image, depth)

    assert out_img.shape == (3, 240, 320)
    assert out_depth.shape == (1, 240, 320)
    assert out_img.dtype == torch.float32
    assert out_depth.dtype == torch.float32
    assert torch.isfinite(out_img).all()
    assert torch.isfinite(out_depth).all()
    assert 0.0 <= out_depth.min().item() <= out_depth.max().item() <= 1.0

    resized_img = TF.resize(
        image, [240, 320], interpolation=InterpolationMode.BILINEAR, antialias=True
    )
    expected_img = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(resized_img)
    assert torch.allclose(out_img, expected_img, atol=1e-6)


def test_train_transform_synchronized_hflip():
    image = torch.zeros(3, 64, 80, dtype=torch.float32)
    image[:, :, :40] = 0.2
    image[:, :, 40:] = 0.8

    depth = torch.zeros(1, 64, 80, dtype=torch.float32)
    depth[:, :, :40] = 0.2
    depth[:, :, 40:] = 0.8

    no_flip = TrainTransform(size=(64, 80), flip_prob=0.0, color_jitter=False)
    img_no_flip, depth_no_flip = no_flip(image, depth)
    assert torch.allclose(denormalize_image(img_no_flip), image, atol=1e-6)
    assert torch.allclose(depth_no_flip, depth, atol=1e-6)

    always_flip = TrainTransform(size=(64, 80), flip_prob=1.0, color_jitter=False)
    img_flip, depth_flip = always_flip(image, depth)
    assert torch.allclose(denormalize_image(img_flip), torch.flip(image, dims=[2]), atol=1e-6)
    assert torch.allclose(depth_flip, torch.flip(depth, dims=[2]), atol=1e-6)


def test_train_transform_color_jitter_preserves_depth():
    torch.manual_seed(0)
    image = torch.rand(3, 120, 160, dtype=torch.float32)
    depth = torch.rand(1, 120, 160, dtype=torch.float32)

    train_tf = TrainTransform(size=(64, 80), flip_prob=0.0, color_jitter=True)
    eval_tf = EvalTransform(size=(64, 80))

    train_img, train_depth = train_tf(image, depth)
    _, eval_depth = eval_tf(image, depth)

    assert torch.isfinite(train_img).all()
    assert torch.allclose(train_depth, eval_depth, atol=1e-6)


def test_denormalize_image_3d_and_4d():
    image_3d = torch.rand(3, 64, 80, dtype=torch.float32)
    depth_3d = torch.rand(1, 64, 80, dtype=torch.float32)

    eval_tf = EvalTransform(size=(64, 80))
    norm_3d, _ = eval_tf(image_3d, depth_3d)
    assert torch.allclose(denormalize_image(norm_3d), image_3d, atol=1e-5)

    image_4d = torch.stack([image_3d, torch.flip(image_3d, dims=[2])], dim=0)
    norm_4d = torch.stack(
        [eval_tf(image_4d[0], depth_3d)[0], eval_tf(image_4d[1], depth_3d)[0]],
        dim=0,
    )
    assert torch.allclose(denormalize_image(norm_4d), image_4d, atol=1e-5)

    with pytest.raises(ValueError):
        denormalize_image(torch.rand(64, 80))


def test_transforms_integrate_with_get_dataloaders(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "split.csv"

    rows = []
    for i in range(3):
        rgb = torch.randint(0, 256, (3, 100, 120), dtype=torch.uint8)
        depth = torch.randint(0, 256, (1, 100, 120), dtype=torch.uint8)
        rgb_path = images_dir / f"rgb_{i}.png"
        depth_path = images_dir / f"depth_{i}.png"
        write_png(rgb, str(rgb_path))
        write_png(depth, str(depth_path))
        rows.append((str(rgb_path), str(depth_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=csv_path,
        val_csv=csv_path,
        test_csv=csv_path,
        batch_size=2,
        train_transform=TrainTransform(size=(64, 80)),
        eval_transform=EvalTransform(size=(64, 80)),
    )

    for loader in (train_loader, val_loader, test_loader):
        batch_img, batch_depth = next(iter(loader))
        assert batch_img.shape == (2, 3, 64, 80)
        assert batch_depth.shape == (2, 1, 64, 80)
        assert torch.isfinite(batch_img).all()
        assert 0.0 <= batch_depth.min().item() <= batch_depth.max().item() <= 1.0
