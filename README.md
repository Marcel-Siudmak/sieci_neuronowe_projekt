# Monocular Image Depth Estimation (NYU Depth V2)

Projekt sieci neuronowej do estymacji mapy głębi z pojedynczego obrazu RGB (`monocular depth estimation`) w PyTorch, wytrenowanej na zbiorze **NYU Depth V2**.

## Zbiór danych i podział

Dane pochodzą ze zbioru NYU Depth V2 (`sohaibanwaar1203/image-depth-estimation` z Kaggle). Skrypt [scripts/prepare_data.py](scripts/prepare_data.py) weryfikuje integralność par RGB+Depth, usuwa duplikaty klatek (`73` nadmiarowe klatki) oraz wykonuje **scene-aware split** na poziomie bazowych pomieszczeń (brak wycieku ujęć z tego samego pokoju między zbiorami):

- **Razem**: `50 615` unikalnych próbek w `249` pomieszczeniach
- **Train**: `35 437` próbek (`70.01%`, `131` pomieszczeń)
- **Val**: `7 590` próbek (`15.00%`, `59` pomieszczeń)
- **Test**: `7 588` próbek (`14.99%`, `59` pomieszczeń)

Wartości map głębi w `nyu2_train` są znormalizowane do przedziału `[0.0, 1.0]`, co odpowiada zakresowi `0–10 m` (`MAX_DEPTH_METERS = 10.0`).

## Architektura i funkcja straty

- **Model (`DepthUNet`)** w [src/model.py](src/model.py): enkoder typu ResNet (`resnet18` lub `resnet34` z wagami ImageNet) połączony z dekoderem U-Net (`UpBlock`: interpolacja bilinear do rozmiaru skip-connection + dwie warstwy `Conv2d 3×3 / BatchNorm2d / ReLU`) oraz głowicą `Sigmoid` na wyjściu `(B, 1, H, W)` w `[0, 1]`.
- **Funkcja straty (`DepthLoss`)** w [src/loss.py](src/loss.py): suma straty pikselowej `L1` oraz straty gradientowej (`L1` na różnicach w osiach `x` i `y` ważone współczynnikiem `lambda_grad`).

## Struktura projektu

```text
├── scripts/
│   ├── download_dataset.py   # pobranie datasetu z Kaggle
│   ├── prepare_data.py       # czyszczenie, deduplikacja i podział scene-aware
│   ├── train.py              # pętla treningowa (CUDA / MPS / CPU)
│   └── evaluate.py           # ewaluacja na test.csv i predykcja pojedynczego obrazu
├── src/
│   ├── data_loader.py        # DepthDataset i get_dataloaders
│   ├── transforms.py         # TrainTransform, EvalTransform, denormalize_image
│   ├── model.py              # DepthUNet
│   ├── loss.py               # DepthLoss (L1 + GradientLoss)
│   └── metrics.py            # MAE, RMSE [m], AbsRel, delta1
└── tests/                    # testy jednostkowe i smoke testy (pytest)
```

## Szybki start

### 1. Instalacja zależności

```bash
uv sync
```

### 2. Pobranie i przygotowanie danych

```bash
uv run python scripts/download_dataset.py
uv run python scripts/prepare_data.py
```

### 3. Uruchomienie testów

```bash
uv run pytest -v
```

### 4. Trening modelu

**Szybki prototyp na MacBooku (Apple Silicon / MPS):**

```bash
uv run python scripts/train.py --backbone resnet18 --img-size 240 320 --batch-size 16 --epochs 3 --max-samples 1000
```

**Pełny trening na GPU NVIDIA (CUDA):**

```bash
uv run python scripts/train.py --backbone resnet34 --img-size 480 640 --batch-size 16 --epochs 15 --num-workers 4
```

Najlepszy checkpoint zapisywany jest do `checkpoints/best.pt`, a wykres strat do `outputs/loss_curve.png`.

### 5. Ewaluacja na `test.csv` i predykcja

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --test-csv dataset/splits/test.csv
```

Wyniki zapisują się do `outputs/test_metrics.json`, a siatka porównawcza (`RGB | Ground Truth | Predykcja`) do `outputs/predictions.png`.

Predykcja na pojedynczym zdjęciu (np. z `dataset/demo/img_nyu2.png`):

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --image dataset/demo/img_nyu2.png --output outputs/demo_pred.png
```

## Metryki i wyniki

Ewaluacja raportuje cztery standardowe metryki (w skali metrycznej `0–10 m`): `MAE [m]` ($\downarrow$), `RMSE [m]` ($\downarrow$), `AbsRel` ($\downarrow$) oraz `delta1` ($\delta < 1.25$, $\uparrow$).

| Konfiguracja | Rozdzielczość | Epoki | MAE [m] | RMSE [m] | AbsRel | delta1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `resnet18` (prototyp MPS) | `240×320` | - | - | - | - | - |
| `resnet34` (docelowy CUDA) | `480×640` | - | - | - | - | - |
