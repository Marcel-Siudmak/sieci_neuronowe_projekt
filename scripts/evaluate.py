import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import ImageReadMode, read_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import get_device, validate
from src.data_loader import DepthDataset
from src.loss import DepthLoss
from src.metrics import MAX_DEPTH_METERS
from src.model import DepthUNet
from src.transforms import EvalTransform, denormalize_image


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[DepthUNet, Tuple[int, int]]:
    """Wczytuje model DepthUNet oraz docelowy rozmiar obrazu z pliku checkpointu."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone = ckpt["backbone"]
    img_size = (int(ckpt["img_size"][0]), int(ckpt["img_size"][1]))

    model = DepthUNet(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, img_size


def save_test_predictions_grid(
    model: DepthUNet,
    dataset: Dataset,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4,
) -> None:
    """Zapisuje siatkę porównawczą RGB | Ground Truth | Predykcja dla wybranych próbek."""
    k = min(num_samples, len(dataset))
    if k <= 0:
        return

    indices = np.linspace(0, len(dataset) - 1, k, dtype=int)
    fig, axes = plt.subplots(k, 3, figsize=(12, 3.5 * k), squeeze=False)

    model.eval()
    for row_idx, ds_idx in enumerate(indices):
        image, depth = dataset[int(ds_idx)]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()

        rgb_np = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
        gt_m = depth.squeeze(0).cpu().numpy() * MAX_DEPTH_METERS
        pred_m = pred.squeeze(0).numpy() * MAX_DEPTH_METERS

        vmin = float(min(gt_m.min(), pred_m.min()))
        vmax = max(float(max(gt_m.max(), pred_m.max())), vmin + 1e-3)

        axes[row_idx, 0].imshow(rgb_np)
        axes[row_idx, 1].imshow(gt_m, cmap="magma", vmin=vmin, vmax=vmax)
        im_pred = axes[row_idx, 2].imshow(pred_m, cmap="magma", vmin=vmin, vmax=vmax)

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

        if row_idx == 0:
            axes[0, 0].set_title("RGB")
            axes[0, 1].set_title("Ground Truth [m]")
            axes[0, 2].set_title("Predykcja [m]")

        fig.colorbar(im_pred, ax=axes[row_idx, 2], fraction=0.046, pad=0.04, label="m")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def predict_single_image(
    model: DepthUNet,
    image_path: Path,
    img_size: Tuple[int, int],
    device: torch.device,
    save_path: Path,
) -> dict:
    """Wykonuje predykcję mapy głębi dla pojedynczego obrazu RGB i zapisuje wizualizację."""
    image_raw = read_image(str(image_path), mode=ImageReadMode.RGB).float() / 255.0
    dummy_depth = torch.zeros(1, image_raw.shape[1], image_raw.shape[2], dtype=torch.float32)
    image_tensor, _ = EvalTransform(size=img_size)(image_raw, dummy_depth)

    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

    rgb_np = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()
    pred_m = pred.squeeze(0).numpy() * MAX_DEPTH_METERS
    vmin = float(pred_m.min())
    vmax = max(float(pred_m.max()), vmin + 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    im_pred = axes[0, 1].imshow(pred_m, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Predykcja głębi [m]")
    axes[0, 1].axis("off")
    fig.colorbar(im_pred, ax=axes[0, 1], fraction=0.046, pad=0.04, label="m")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    result = {
        "image": str(image_path),
        "output": str(save_path),
        "min_depth_m": float(pred_m.min()),
        "max_depth_m": float(pred_m.max()),
    }
    print(
        f"Predykcja zapisana do {save_path} "
        f"(zakres głębi: {result['min_depth_m']:.2f}m .. {result['max_depth_m']:.2f}m)"
    )
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ewaluacja modelu DepthUNet i wizualizacja predykcji.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--test-csv", type=Path, default=Path("dataset/splits/test.csv"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-vis", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.png"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/test_metrics.json"))
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict:
    """Przeprowadza ewaluację na zbiorze testowym lub predykcję na pojedynczym obrazie."""
    device = get_device()
    model, img_size = load_model_from_checkpoint(args.checkpoint, device)

    if args.image is not None:
        return predict_single_image(model, args.image, img_size, device, args.output)

    test_ds: Dataset = DepthDataset(args.test_csv, transform=EvalTransform(size=img_size))
    if args.max_samples is not None and args.max_samples < len(test_ds):
        test_ds = Subset(test_ds, range(args.max_samples))

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    test_loss, metrics = validate(model, test_loader, DepthLoss(), device)
    results = {
        "num_samples": len(test_ds),
        "test_loss": float(test_loss),
        **metrics,
    }

    print("\n=== Wyniki na zbiorze testowym ===")
    print(f"Próbki:    {results['num_samples']}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"MAE:       {results['mae']:.3f} m")
    print(f"RMSE:      {results['rmse']:.3f} m")
    print(f"AbsRel:    {results['abs_rel']:.3f}")
    print(f"delta1:    {results['delta1']:.3f}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    save_test_predictions_grid(model, test_ds, device, args.output, num_samples=args.num_vis)
    print(f"Zapisano metryki: {args.metrics_out}")
    print(f"Zapisano wizualizację: {args.output}")
    return results


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
