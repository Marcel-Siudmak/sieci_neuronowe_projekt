import argparse
import math
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataloaders
from src.loss import DepthLoss
from src.metrics import compute_metrics
from src.model import DepthUNet
from src.transforms import EvalTransform, TrainTransform


def get_device() -> torch.device:
    """Wybiera najlepsze dostępne urządzenie obliczeniowe (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Ustawia ziarno losowości dla powtarzalności treningu."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_subset_loader(
    loader: DataLoader,
    max_samples: Optional[int],
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Ogranicza DataLoader do równomiernie rozłożonych `max_samples` próbek (do szybkich testów)."""
    if max_samples is None or max_samples >= len(loader.dataset):
        return loader

    indices = np.linspace(0, len(loader.dataset) - 1, max_samples, dtype=int).tolist()
    subset = Subset(loader.dataset, indices)
    batch_size = loader.batch_size or 1
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last and len(subset) >= batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Przeprowadza jedną epokę treningu i zwraca średnią stratę."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = device.type == "cuda" and scaler is not None

    for images, depths in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        depths = depths.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images)
            loss = criterion(preds, depths)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, dict[str, float]]:
    """Ewaluuje model na zbiorze walidacyjnym i zwraca (średnia strata, metryki)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_abs_rel = 0.0
    total_delta1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, depths in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=device.type == "cuda")
            depths = depths.to(device, non_blocking=device.type == "cuda")

            preds = model(images)
            loss = criterion(preds, depths)
            metrics = compute_metrics(preds, depths)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += metrics["mae"] * batch_size
            total_mse += (metrics["rmse"] ** 2) * batch_size
            total_abs_rel += metrics["abs_rel"] * batch_size
            total_delta1 += metrics["delta1"] * batch_size
            total_samples += batch_size

    denom = max(total_samples, 1)
    avg_metrics = {
        "mae": total_mae / denom,
        "rmse": math.sqrt(total_mse / denom),
        "abs_rel": total_abs_rel / denom,
        "delta1": total_delta1 / denom,
    }
    return total_loss / denom, avg_metrics


def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
) -> None:
    """Zapisuje wykres przebiegu straty treningowej i walidacyjnej."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs, val_losses, marker="o", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening modelu DepthUNet na zbiorze NYU Depth V2.")
    parser.add_argument("--train-csv", type=Path, default=Path("dataset/splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("dataset/splits/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("dataset/splits/test.csv"))
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--no-pretrained", action="store_true", default=False)
    parser.add_argument("--img-size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-grad", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/loss_curve.png"))
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict:
    """Uruchamia pełną pętlę treningową i zwraca historię strat."""
    set_seed(args.seed)
    device = get_device()
    img_size = (int(args.img_size[0]), int(args.img_size[1]))

    train_loader, val_loader, _ = get_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        train_transform=TrainTransform(size=img_size),
        eval_transform=EvalTransform(size=img_size),
    )
    train_loader = _maybe_subset_loader(train_loader, args.max_samples, shuffle=True, drop_last=True)
    val_loader = _maybe_subset_loader(val_loader, args.max_samples, shuffle=False, drop_last=False)

    model = DepthUNet(backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    criterion = DepthLoss(lambda_grad=args.lambda_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(
        f"Start treningu: device={device.type}, backbone={args.backbone}, "
        f"img_size={img_size}, train_samples={len(train_loader.dataset)}, "
        f"val_samples={len(val_loader.dataset)}"
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE={val_metrics['mae']:.3f}m | RMSE={val_metrics['rmse']:.3f}m | "
            f"AbsRel={val_metrics['abs_rel']:.3f} | delta1={val_metrics['delta1']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "img_size": img_size,
                    "model_state_dict": model.state_dict(),
                    "val_loss": float(val_loss),
                    "val_metrics": val_metrics,
                },
                args.checkpoint,
            )

        save_loss_curve(train_losses, val_losses, args.plot_path)

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
