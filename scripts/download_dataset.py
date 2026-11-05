from pathlib import Path
import kagglehub

# Ścieżka do folderu 'dataset' w katalogu głównym projektu
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset"

# Pobranie najnowszej wersji datasetu do folderu docelowego
path = kagglehub.dataset_download(
    "sohaibanwaar1203/image-depth-estimation",
    output_dir=str(OUTPUT_DIR),
    force_download=False,
)

print("Path to dataset files:", path)

