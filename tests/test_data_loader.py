import csv
from pathlib import Path
import pytest
import torch
from torchvision.io import write_png
from src.data_loader import DepthDataset, get_dataloaders


@pytest.fixture
def mock_dataset_dir(tmp_path: Path):
    """Tworzy tymczasowe środowisko z syntetycznymi danymi RGB i głębi."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Wygeneruj 5 przykładowych par obrazów o wymiarach 100x120
    samples = []
    for i in range(5):
        rgb_tensor = torch.randint(0, 256, (3, 100, 120), dtype=torch.uint8)
        depth_tensor = torch.randint(0, 256, (1, 100, 120), dtype=torch.uint8)

        rgb_path = images_dir / f"rgb_{i}.png"
        depth_path = images_dir / f"depth_{i}.png"

        write_png(rgb_tensor, str(rgb_path))
        write_png(depth_tensor, str(depth_path))

        samples.append((str(rgb_path), str(depth_path)))

    # CSV bez nagłówka
    csv_no_header = tmp_path / "data_no_header.csv"
    with open(csv_no_header, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for rgb_p, depth_p in samples:
            writer.writerow([rgb_p, depth_p])

    # CSV z nagłówkiem
    csv_with_header = tmp_path / "data_with_header.csv"
    with open(csv_with_header, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "depth"])
        for rgb_p, depth_p in samples:
            writer.writerow([rgb_p, depth_p])

    return {
        "dir": tmp_path,
        "csv_no_header": str(csv_no_header),
        "csv_with_header": str(csv_with_header),
        "num_samples": len(samples),
        "samples": samples,
    }


def test_dataset_init_and_len_no_header(mock_dataset_dir):
    """Testuje inicjalizację i poprawność długości dla pliku CSV bez nagłówka."""
    ds = DepthDataset(mock_dataset_dir["csv_no_header"])
    assert len(ds) == mock_dataset_dir["num_samples"]


def test_dataset_init_and_len_with_header(mock_dataset_dir):
    """Testuje inicjalizację i poprawność długości dla pliku CSV z nagłówkiem."""
    ds = DepthDataset(mock_dataset_dir["csv_with_header"])
    assert len(ds) == mock_dataset_dir["num_samples"]


def test_dataset_getitem_output(mock_dataset_dir):
    """Testuje format, wymiary i normalizację danych do [0.0, 1.0]."""
    ds = DepthDataset(mock_dataset_dir["csv_no_header"])
    image, depth = ds[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(depth, torch.Tensor)

    # Zachowanie wymiarów
    assert image.shape == (3, 100, 120)
    assert depth.shape == (1, 100, 120)

    # Typ danych float32
    assert image.dtype == torch.float32
    assert depth.dtype == torch.float32

    # Normalizacja do [0.0, 1.0]
    assert 0.0 <= image.min().item() <= 1.0
    assert 0.0 <= image.max().item() <= 1.0
    assert 0.0 <= depth.min().item() <= 1.0
    assert 0.0 <= depth.max().item() <= 1.0


def test_dataset_transform(mock_dataset_dir):
    """Testuje aplikowanie wspólnej transformacji RGB i głębi."""
    def mock_flip(img: torch.Tensor, depth: torch.Tensor):
        return torch.flip(img, dims=[2]), torch.flip(depth, dims=[2])

    ds_original = DepthDataset(mock_dataset_dir["csv_no_header"])
    ds_flipped = DepthDataset(mock_dataset_dir["csv_no_header"], transform=mock_flip)

    orig_img, orig_depth = ds_original[0]
    flip_img, flip_depth = ds_flipped[0]

    assert torch.allclose(flip_img, torch.flip(orig_img, dims=[2]))
    assert torch.allclose(flip_depth, torch.flip(orig_depth, dims=[2]))


def test_dataset_file_not_found(tmp_path: Path):
    """Testuje rzucanie błędu przy brakującym pliku CSV lub obrazie."""
    with pytest.raises(FileNotFoundError):
        _ = DepthDataset(str(tmp_path / "non_existent.csv"))

    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["non_existent_rgb.png", "non_existent_depth.png"])

    ds = DepthDataset(str(bad_csv))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        _ = ds[0]


def test_dataset_index_out_of_bounds(mock_dataset_dir):
    """Testuje, czy próba odpytania o błędny indeks rzuca IndexError."""
    ds = DepthDataset(mock_dataset_dir["csv_no_header"])
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_get_dataloaders(mock_dataset_dir):
    """Testuje tworzenie i działanie DataLoaderów (train, val, test) z parametrem drop_last."""
    csv_file = mock_dataset_dir["csv_no_header"]
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=csv_file,
        val_csv=csv_file,
        test_csv=csv_file,
        batch_size=2,
    )

    batches_train = list(train_loader)
    assert len(batches_train) == 2  # 5 // 2 = 2 (drop_last=True)

    batches_val = list(val_loader)
    assert len(batches_val) == 3  # 5 próbek: 2, 2, 1 (drop_last=False)

    batch_img, batch_depth = batches_train[0]
    assert batch_img.shape == (2, 3, 100, 120)
    assert batch_depth.shape == (2, 1, 100, 120)
    assert batch_img.dtype == torch.float32
    assert batch_depth.dtype == torch.float32


def test_real_dataset_if_present():
    """Weryfikuje działanie na rzeczywistych danych projektu."""
    train_csv = Path("dataset/splits/train.csv")
    if not train_csv.exists():
        pytest.skip("dataset/splits/train.csv nie istnieje w środowisku.")

    ds = DepthDataset(str(train_csv))
    assert len(ds) > 0

    image, depth = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(depth, torch.Tensor)
    assert image.shape == (3, 480, 640)
    assert depth.shape == (1, 480, 640)
    assert image.dtype == torch.float32
    assert depth.dtype == torch.float32
