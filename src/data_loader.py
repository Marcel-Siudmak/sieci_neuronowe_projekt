import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


class DepthDataset(Dataset):
    """Dataset wczytujący pary obraz RGB oraz mapa głębi z pliku CSV."""

    def __init__(
        self,
        csv_file: Union[str, Path],
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.samples: list[Tuple[str, str]] = []

        with open(self.csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().lower() in ("image", "rgb", "img", "image_path"):
                    continue
                self.samples.append((row[0].strip(), row[1].strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.samples[index]

        image = read_image(rgb_path, mode=ImageReadMode.RGB).float() / 255.0
        depth = read_image(depth_path, mode=ImageReadMode.GRAY).float() / 255.0

        if self.transform is not None:
            image, depth = self.transform(image, depth)

        return image, depth


def get_dataloaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoadery dla zbiorów treningowego, walidacyjnego i testowego."""
    train_loader = DataLoader(
        DepthDataset(train_csv, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        DepthDataset(val_csv),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        DepthDataset(test_csv),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
