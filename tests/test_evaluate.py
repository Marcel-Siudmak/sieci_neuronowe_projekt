import csv
import json
import math
from pathlib import Path

import torch
from torchvision.io import write_png

from scripts.evaluate import parse_args, run_evaluation
from src.model import DepthUNet


def _save_dummy_checkpoint(path: Path, img_size=(64, 80)) -> None:
    model = DepthUNet(backbone="resnet18", pretrained=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 1,
            "backbone": "resnet18",
            "img_size": img_size,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.5,
            "val_metrics": {"mae": 1.0, "rmse": 1.2, "abs_rel": 0.2, "delta1": 0.5},
        },
        path,
    )


def test_evaluate_on_test_csv(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    test_csv = tmp_path / "test.csv"

    rows = []
    for i in range(4):
        rgb = torch.randint(0, 256, (3, 80, 100), dtype=torch.uint8)
        depth = torch.randint(20, 240, (1, 80, 100), dtype=torch.uint8)
        rgb_path = images_dir / f"rgb_{i}.png"
        depth_path = images_dir / f"depth_{i}.png"
        write_png(rgb, str(rgb_path))
        write_png(depth, str(depth_path))
        rows.append((str(rgb_path), str(depth_path)))

    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    ckpt_path = tmp_path / "best.pt"
    out_img = tmp_path / "outputs" / "predictions.png"
    metrics_json = tmp_path / "outputs" / "test_metrics.json"
    _save_dummy_checkpoint(ckpt_path)

    args = parse_args(
        [
            "--checkpoint",
            str(ckpt_path),
            "--test-csv",
            str(test_csv),
            "--batch-size",
            "2",
            "--num-vis",
            "2",
            "--output",
            str(out_img),
            "--metrics-out",
            str(metrics_json),
        ]
    )
    results = run_evaluation(args)

    assert results["num_samples"] == 4
    for key in ("test_loss", "mae", "rmse", "abs_rel", "delta1"):
        assert math.isfinite(results[key])

    assert out_img.exists() and out_img.stat().st_size > 0
    assert metrics_json.exists()
    saved = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert set(saved.keys()) == {"num_samples", "test_loss", "mae", "rmse", "abs_rel", "delta1"}


def test_evaluate_single_image(tmp_path: Path):
    rgb = torch.randint(0, 256, (3, 120, 160), dtype=torch.uint8)
    img_path = tmp_path / "sample.png"
    write_png(rgb, str(img_path))

    ckpt_path = tmp_path / "best.pt"
    out_img = tmp_path / "outputs" / "single_pred.png"
    _save_dummy_checkpoint(ckpt_path)

    args = parse_args(
        [
            "--checkpoint",
            str(ckpt_path),
            "--image",
            str(img_path),
            "--output",
            str(out_img),
        ]
    )
    result = run_evaluation(args)

    assert out_img.exists() and out_img.stat().st_size > 0
    assert 0.0 <= result["min_depth_m"] <= result["max_depth_m"] <= 10.0
