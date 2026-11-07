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

    # Wygeneruj 5 przykładowych par obrazów o wymiarach 100x120 (aby zweryfikować zachowanie rozmiarów)
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
    assert len(ds) == mock_dataset_dir["num_samples"], (
        f"Oczekiwano {mock_dataset_dir['num_samples']} próbek, otrzymano {len(ds)}"
    )


def test_dataset_init_and_len_with_header(mock_dataset_dir):
    """Testuje inicjalizację i poprawność długości dla pliku CSV z nagłówkiem."""
    ds = DepthDataset(mock_dataset_dir["csv_with_header"])
    assert len(ds) == mock_dataset_dir["num_samples"], (
        f"Oczekiwano {mock_dataset_dir['num_samples']} próbek, nagłówek nie powinien być liczony jako próbka"
    )


def test_dataset_getitem_output_and_no_downscaling(mock_dataset_dir):
    """Testuje format, brak downscalingu (zachowanie rozmiaru 100x120) i normalizację."""
    ds = DepthDataset(mock_dataset_dir["csv_no_header"])
    image, depth = ds[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(depth, torch.Tensor)

    # Weryfikacja braku downscalingu - zachowanie oryginalnych wymiarów 100x120
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


def test_dataset_joint_transform(mock_dataset_dir):
    """Testuje synchroniczną transformację RGB i głębi (np. poziome odbicie lustrzane)."""
    def mock_joint_flip(img: torch.Tensor, depth: torch.Tensor):
        return torch.flip(img, dims=[2]), torch.flip(depth, dims=[2])

    ds_original = DepthDataset(mock_dataset_dir["csv_no_header"])
    ds_flipped = DepthDataset(mock_dataset_dir["csv_no_header"], joint_transform=mock_joint_flip)

    orig_img, orig_depth = ds_original[0]
    flip_img, flip_depth = ds_flipped[0]

    assert torch.allclose(flip_img, torch.flip(orig_img, dims=[2]))
    assert torch.allclose(flip_depth, torch.flip(orig_depth, dims=[2]))


def test_dataset_transforms(mock_dataset_dir):
    """Testuje aplikowanie niezależnych transformacji dla RGB i docelowej głębi."""
    rgb_tf = lambda img: img * 0.5
    depth_tf = lambda d: d * 0.25
    ds = DepthDataset(
        mock_dataset_dir["csv_no_header"],
        transform=rgb_tf,
        target_transform=depth_tf,
    )
    image, depth = ds[0]
    assert image.max().item() <= 0.5 + 1e-5
    assert depth.max().item() <= 0.25 + 1e-5


def test_dataset_with_root_dir(tmp_path: Path):
    """Testuje działanie parametru root_dir z relatywnymi ścieżkami w CSV."""
    sub_dir = tmp_path / "dataset"
    sub_dir.mkdir()

    rgb_tensor = torch.zeros((3, 32, 32), dtype=torch.uint8)
    depth_tensor = torch.zeros((1, 32, 32), dtype=torch.uint8)

    rgb_file = sub_dir / "img.png"
    depth_file = sub_dir / "depth.png"
    write_png(rgb_tensor, str(rgb_file))
    write_png(depth_tensor, str(depth_file))

    csv_path = tmp_path / "rel.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img.png", "depth.png"])

    ds = DepthDataset(str(csv_path), root_dir=str(sub_dir))
    assert len(ds) == 1
    image, depth = ds[0]
    assert image.shape == (3, 32, 32)
    assert depth.shape == (1, 32, 32)


def test_dataset_file_not_found(tmp_path: Path):
    """Testuje rzucanie FileNotFoundError przy brakującym pliku CSV lub brakującym obrazie."""
    with pytest.raises(FileNotFoundError):
        _ = DepthDataset(str(tmp_path / "non_existent.csv"))

    # CSV ze ścieżką do nieistniejącego pliku obrazu
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["non_existent_rgb.png", "non_existent_depth.png"])

    ds = DepthDataset(str(bad_csv))
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_get_dataloaders(mock_dataset_dir):
    """Testuje tworzenie i działanie DataLoaderów (train, val, test) z parametrem drop_last."""
    csv_file = mock_dataset_dir["csv_no_header"]
    # 5 próbek, batch_size=2 -> z drop_last=True w train_loaderze będą 2 batche (4 próbki), a w val_loaderze 3 batche
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=csv_file,
        val_csv=csv_file,
        test_csv=csv_file,
        batch_size=2,
        drop_last=True,
    )

    batches_train = list(train_loader)
    assert len(batches_train) == 2  # 5 // 2 = 2 batche (ostatnia pojedyncza próbka odrzucona dla BatchNorm)

    batches_val = list(val_loader)
    assert len(batches_val) == 3  # w ewaluacji nie odrzucamy danych (5 próbek: 2, 2, 1)

    batch_img, batch_depth = batches_train[0]
    assert batch_img.shape == (2, 3, 100, 120)
    assert batch_depth.shape == (2, 1, 100, 120)
    assert batch_img.dtype == torch.float32
    assert batch_depth.dtype == torch.float32


def test_dataset_index_out_of_bounds(mock_dataset_dir):
    """Testuje, czy próba odpytania o błędny indeks rzuca IndexError."""
    ds = DepthDataset(mock_dataset_dir["csv_no_header"])
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-1]


def test_real_dataset_if_present():
    """Weryfikuje działanie na rzeczywistych danych projektu (pełna rozdzielczość 480x640)."""
    train_csv = Path("dataset/splits/train.csv")
    if not train_csv.exists():
        pytest.skip("dataset/splits/train.csv nie istnieje w środowisku.")

    ds = DepthDataset(str(train_csv))
    assert len(ds) > 0

    image, depth = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(depth, torch.Tensor)
    # Pełna, oryginalna rozdzielczość NYUv2 zachowana bez downscalingu
    assert image.shape == (3, 480, 640)
    assert depth.shape == (1, 480, 640)
    assert image.dtype == torch.float32
    assert depth.dtype == torch.float32
