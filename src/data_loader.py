import csv
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import warnings

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image

# Wyciszenie deprecation warning ze starszych dekoderów torchvision
warnings.filterwarnings(
    "ignore",
    message=".*The image decoding and encoding capabilities of TorchVision are deprecated.*",
    category=DeprecationWarning,
)


def _is_header(row: list[str]) -> bool:
    """Sprawdza, czy wiersz pliku CSV jest nagłówkiem."""
    if not row or len(row) < 2:
        return False
    first_col = row[0].strip().lower()
    if first_col in ("image", "rgb", "img", "rgb_path", "image_path", "color"):
        return True
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return not any(first_col.endswith(ext) for ext in valid_exts)


class DepthDataset(Dataset):
    """
    Dataset wczytujący pary obraz RGB oraz odpowiadającą mapę głębi z pliku CSV.
    Zachowuje oryginalną rozdzielczość obrazów (brak wymuszonego downscalingu).
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        joint_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Plik CSV nie istnieje: {self.csv_file}")

        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.samples: list[Tuple[str, str]] = []

        with open(self.csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if _is_header(row):
                    continue
                self.samples.append((row[0].strip(), row[1].strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Indeks {index} poza zakresem datasetu o rozmiarze {len(self.samples)}.")

        rgb_path, depth_path = self.samples[index]

        if self.root_dir is not None:
            rgb_path = str(self.root_dir / rgb_path)
            depth_path = str(self.root_dir / depth_path)

        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(f"Nie znaleziono pliku obrazu RGB: {rgb_path}")
        if not os.path.isfile(depth_path):
            raise FileNotFoundError(f"Nie znaleziono pliku mapy głębi: {depth_path}")

        # Odczyt obrazów: RGB (3 kanały) oraz mapa głębi (1 kanał w skali szarości)
        # Zachowujemy oryginalny rozmiar (H, W) bez downscalingu
        image = read_image(rgb_path, mode=ImageReadMode.RGB)
        depth = read_image(depth_path, mode=ImageReadMode.GRAY)

        # Normalizacja wartości pikseli z [0, 255] do [0.0, 1.0]
        image = image.float() / 255.0
        depth = depth.float() / 255.0

        # Wspólne przekształcenia przestrzenne (np. synchroniczny losowy flip/crop)
        if self.joint_transform is not None:
            image, depth = self.joint_transform(image, depth)

        # Dedykowane transformacje obrazu RGB (np. ColorJitter, normalizacja barw ImageNet)
        if self.transform is not None:
            image = self.transform(image)

        # Dedykowane transformacje mapy głębi
        if self.target_transform is not None:
            depth = self.target_transform(depth)

        return image, depth


def get_dataloaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    root_dir: Optional[Union[str, Path]] = None,
    joint_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    val_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    val_target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tworzy DataLoadery dla zbiorów treningowego, walidacyjnego i testowego.
    
    Parametr drop_last=True dla zbioru treningowego zabezpiecza warstwy BatchNorm2d
    przed błędem pojedynczej próbki w ostatnim batchu epoki.
    """
    train_dataset = DepthDataset(
        train_csv,
        root_dir=root_dir,
        joint_transform=joint_transform,
        transform=transform,
        target_transform=target_transform,
    )
    val_dataset = DepthDataset(
        val_csv,
        root_dir=root_dir,
        transform=val_transform if val_transform is not None else transform,
        target_transform=val_target_transform if val_target_transform is not None else target_transform,
    )
    test_dataset = DepthDataset(
        test_csv,
        root_dir=root_dir,
        transform=val_transform if val_transform is not None else transform,
        target_transform=val_target_transform if val_target_transform is not None else target_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
