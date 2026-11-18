import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import write_png

from scripts.train import _maybe_subset_loader, parse_args, run_training


def test_maybe_subset_loader_spreads_indices():
    dataset = TensorDataset(torch.arange(100))
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    subset_loader = _maybe_subset_loader(loader, max_samples=10, shuffle=False, drop_last=False)
    indices = subset_loader.dataset.indices

    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 99
    assert indices != list(range(10))


def test_training_smoke_run(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "split.csv"

    rows = []
    for i in range(4):
        rgb = torch.randint(0, 256, (3, 80, 100), dtype=torch.uint8)
        depth = torch.randint(20, 240, (1, 80, 100), dtype=torch.uint8)
        rgb_path = images_dir / f"rgb_{i}.png"
        depth_path = images_dir / f"depth_{i}.png"
        write_png(rgb, str(rgb_path))
        write_png(depth, str(depth_path))
        rows.append((str(rgb_path), str(depth_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    ckpt_path = tmp_path / "checkpoints" / "best.pt"
    plot_path = tmp_path / "outputs" / "loss_curve.png"

    args = parse_args(
        [
            "--train-csv",
            str(csv_path),
            "--val-csv",
            str(csv_path),
            "--test-csv",
            str(csv_path),
            "--no-pretrained",
            "--img-size",
            "64",
            "80",
            "--batch-size",
            "2",
            "--epochs",
            "2",
            "--checkpoint",
            str(ckpt_path),
            "--plot-path",
            str(plot_path),
        ]
    )

    history = run_training(args)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(math.isfinite(x) for x in history["train_loss"])
    assert all(math.isfinite(x) for x in history["val_loss"])

    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert set(ckpt.keys()) == {
        "epoch",
        "backbone",
        "img_size",
        "model_state_dict",
        "val_loss",
        "val_metrics",
    }
    assert ckpt["backbone"] == "resnet18"
    assert tuple(ckpt["img_size"]) == (64, 80)

    assert plot_path.exists() and plot_path.stat().st_size > 0
