#!/usr/bin/env python3
"""
Skrypt przygotowujący dane do treningu sieci neuronowej (NYU Depth V2).
1. Weryfikuje integralność par obrazów (RGB + mapa głębi) oraz brak uszkodzonych/pustych plików.
2. Sprawdza i raportuje obecność powtórzeń (duplikatów) na poziomie haszy SHA-256 oraz wykrywa przecieki między zbiorami.
3. Dzieli zbiór na train, validation oraz test z uwzględnieniem scen (scene-aware split, brak data leakage) lub wg istniejącego zbioru testowego.
4. Zapisuje gotowe pliki CSV (train.csv, val.csv, test.csv) oraz szczegółowy raport JSON.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Oblicza hash SHA-256 pliku."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_pairs(
    dataset_dir: Path,
) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]]]:
    """
    Wczytuje pary (rgb, depth, scene) dla zbioru treningowego i testowego.
    Zwraca (train_pairs, test_pairs).
    """
    train_dir = dataset_dir / "nyu2_train"
    test_dir = dataset_dir / "nyu2_test"

    train_pairs: List[Tuple[Path, Path, str]] = []
    test_pairs: List[Tuple[Path, Path, str]] = []

    if train_dir.exists():
        for scene_dir in sorted(train_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue
            scene_name = scene_dir.name
            for rgb_file in sorted(scene_dir.glob("*.jpg")):
                depth_file = rgb_file.with_suffix(".png")
                train_pairs.append((rgb_file, depth_file, scene_name))

    if test_dir.exists():
        for rgb_file in sorted(test_dir.glob("*_colors.png")):
            depth_file = rgb_file.parent / rgb_file.name.replace("_colors.png", "_depth.png")
            test_pairs.append((rgb_file, depth_file, "official_test"))

    return train_pairs, test_pairs


def verify_file_integrity(
    pairs: List[Tuple[Path, Path, str]],
) -> Tuple[List[Tuple[Path, Path, str]], List[str], Dict[Path, int]]:
    """
    Sprawdza, czy oba pliki w parze istnieją i czy ich rozmiar jest większy niż 0.
    Zwraca (valid_pairs, errors, cached_sizes).
    """
    valid_pairs: List[Tuple[Path, Path, str]] = []
    errors: List[str] = []
    cached_sizes: Dict[Path, int] = {}

    for rgb_p, depth_p, scene in pairs:
        try:
            rgb_size = rgb_p.stat().st_size
            depth_size = depth_p.stat().st_size
        except FileNotFoundError as e:
            errors.append(f"Nie znaleziono pliku: {e.filename}")
            continue

        if rgb_size == 0:
            errors.append(f"Pusty plik RGB (0 B): {rgb_p}")
            continue
        if depth_size == 0:
            errors.append(f"Pusty plik głębi (0 B): {depth_p}")
            continue

        cached_sizes[rgb_p] = rgb_size
        cached_sizes[depth_p] = depth_size
        valid_pairs.append((rgb_p, depth_p, scene))

    return valid_pairs, errors, cached_sizes


def find_duplicates(
    pairs: List[Tuple[Path, Path, str]],
    cached_sizes: Dict[Path, int],
) -> Tuple[
    Dict[str, List[Path]],
    Dict[str, List[Path]],
    Dict[Tuple[str, str], List[Tuple[Path, Path]]],
    Dict[Path, str],
    Dict[Path, str],
]:
    """
    Wyszukuje duplikaty na podstawie haszy SHA-256.
    Zoptymalizowane: haszuje tylko pliki o identycznych rozmiarach.
    Zwraca:
      - rgb_dupes: {hash: [paths]}
      - depth_dupes: {hash: [paths]}
      - pair_dupes: {(rgb_hash, depth_hash): [(rgb_p, depth_p)]}
      - rgb_hashes: {path: hash_or_unique_id}
      - depth_hashes: {path: hash_or_unique_id}
    """
    # Krok 1: grupowanie po rozmiarze pliku
    rgb_by_size = defaultdict(list)
    depth_by_size = defaultdict(list)

    for rgb_p, depth_p, _ in pairs:
        rgb_by_size[cached_sizes[rgb_p]].append(rgb_p)
        depth_by_size[cached_sizes[depth_p]].append(depth_p)

    # Krok 2: obliczanie hashy dla potencjalnych kolizji rozmiarowych
    rgb_hashes: Dict[Path, str] = {}
    for size, paths in rgb_by_size.items():
        if len(paths) > 1:
            for p in paths:
                rgb_hashes[p] = compute_file_hash(p)
        else:
            rgb_hashes[paths[0]] = f"unique_{size}_{paths[0].name}"

    depth_hashes: Dict[Path, str] = {}
    for size, paths in depth_by_size.items():
        if len(paths) > 1:
            for p in paths:
                depth_hashes[p] = compute_file_hash(p)
        else:
            depth_hashes[paths[0]] = f"unique_{size}_{paths[0].name}"

    # Krok 3: wykrywanie powtórzeń
    rgb_hash_groups = defaultdict(list)
    for p, h in rgb_hashes.items():
        if not h.startswith("unique_"):
            rgb_hash_groups[h].append(p)

    depth_hash_groups = defaultdict(list)
    for p, h in depth_hashes.items():
        if not h.startswith("unique_"):
            depth_hash_groups[h].append(p)

    pair_hash_groups = defaultdict(list)
    for rgb_p, depth_p, _ in pairs:
        pair_key = (rgb_hashes[rgb_p], depth_hashes[depth_p])
        pair_hash_groups[pair_key].append((rgb_p, depth_p))

    rgb_dupes = {h: paths for h, paths in rgb_hash_groups.items() if len(paths) > 1}
    depth_dupes = {h: paths for h, paths in depth_hash_groups.items() if len(paths) > 1}
    pair_dupes = {k: paths for k, paths in pair_hash_groups.items() if len(paths) > 1}

    return rgb_dupes, depth_dupes, pair_dupes, rgb_hashes, depth_hashes


def deduplicate_pairs(
    pairs: List[Tuple[Path, Path, str]],
    rgb_hashes: Dict[Path, str],
    depth_hashes: Dict[Path, str],
    dedup_mode: str = "rgb",
) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]]]:
    """
    Usuwa zduplikowane próbki:
    - 'rgb': zachowuje pierwszą klatkę dla unikalnego obrazu RGB (odrzuca identyczne klatki wideo)
    - 'pair': zachowuje pierwszą klatkę dla identycznej pary (RGB + Depth)
    Zwraca (unique_pairs, removed_pairs).
    """
    seen_keys: Set[str] = set()
    unique_pairs: List[Tuple[Path, Path, str]] = []
    removed_pairs: List[Tuple[Path, Path, str]] = []

    for rgb_p, depth_p, scene in pairs:
        if dedup_mode == "rgb":
            key = rgb_hashes[rgb_p]
        elif dedup_mode == "pair":
            key = f"{rgb_hashes[rgb_p]}_{depth_hashes[depth_p]}"
        else:
            key = str(rgb_p)

        if key in seen_keys:
            removed_pairs.append((rgb_p, depth_p, scene))
        else:
            seen_keys.add(key)
            unique_pairs.append((rgb_p, depth_p, scene))

    return unique_pairs, removed_pairs


def split_by_scenes(
    pairs: List[Tuple[Path, Path, str]],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, Path, str]],
    List[Tuple[Path, Path, str]],
    List[Tuple[Path, Path, str]],
    Dict[str, List[str]],
]:
    """
    Dzieli zbiór danych na train, val i test wg scen (Scene-aware Group Split).
    Zapobiega wyciekowi danych (data leakage), gdyż klatki z tej samej sceny
    trafiają wyłącznie do jednego zbioru.
    Wykorzystuje greedy bin packing dla precyzyjnego dopasowania zadanych proporcji.
    """
    scene_to_pairs = defaultdict(list)
    for rgb_p, depth_p, scene in pairs:
        scene_to_pairs[scene].append((rgb_p, depth_p, scene))

    total_samples = len(pairs)
    train_ratio, val_ratio, test_ratio = ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    targets = [train_ratio * total_samples, val_ratio * total_samples, test_ratio * total_samples]
    bins: List[List[Tuple[Path, Path, str]]] = [[], [], []]
    bin_counts = [0, 0, 0]
    bin_scenes: List[List[str]] = [[], [], []]

    rng = random.Random(seed)
    scenes = list(scene_to_pairs.keys())
    rng.shuffle(scenes)

    # Sortowanie scen malejąco po liczbie próbek dla dokładnego rozkładu (greedy)
    sorted_scenes = sorted(scenes, key=lambda s: len(scene_to_pairs[s]), reverse=True)

    for scene in sorted_scenes:
        items = scene_to_pairs[scene]
        count = len(items)

        # Wybór koszyka, który najbardziej odstaje poniżej celu
        deficits = [targets[i] - bin_counts[i] for i in range(3)]
        best_bin = deficits.index(max(deficits))

        bins[best_bin].extend(items)
        bin_counts[best_bin] += count
        bin_scenes[best_bin].append(scene)

    scene_allocations = {
        "train_scenes": bin_scenes[0],
        "val_scenes": bin_scenes[1],
        "test_scenes": bin_scenes[2],
    }

    return bins[0], bins[1], bins[2], scene_allocations


def split_random(
    pairs: List[Tuple[Path, Path, str]],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]]]:
    """
    Standardowy losowy podział próbek (ramka po ramce).
    Uwaga: może powodować przeciek scen (scene leakage).
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    train_ratio, val_ratio, test_ratio = ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train : n_train + n_val]
    test_data = shuffled[n_train + n_val :]

    return train_data, val_data, test_data


def save_split_csv(
    filepath: Path,
    pairs: List[Tuple[Path, Path, str]],
    relative_to: Optional[Path] = None,
    add_header: bool = False,
) -> None:
    """Zapisuje zbiór do pliku CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if add_header:
            writer.writerow(["image", "depth", "scene"])
        for rgb_p, depth_p, scene in pairs:
            if relative_to:
                try:
                    rgb_str = str(rgb_p.relative_to(relative_to))
                    depth_str = str(depth_p.relative_to(relative_to))
                except ValueError:
                    rgb_str = str(rgb_p)
                    depth_str = str(depth_p)
            else:
                rgb_str = str(rgb_p)
                depth_str = str(depth_p)

            if add_header:
                writer.writerow([rgb_str, depth_str, scene])
            else:
                writer.writerow([rgb_str, depth_str])


def main():
    parser = argparse.ArgumentParser(
        description="Przygotowanie, deduplikacja i podział datasetu NYU Depth V2 na zbiory train/val/test."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Ścieżka do katalogu z danymi (domyślnie: dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/splits"),
        help="Ścieżka do katalogu zapisu plików podziału CSV (domyślnie: dataset/splits)",
    )
    parser.add_argument(
        "--test-source",
        choices=["split", "existing"],
        default="split",
        help=(
            "Źródło zbioru testowego: 'split' (dzieli sceny nyu2_train na train/val/test), "
            "'existing' (używa nyu2_test jako test, a nyu2_train dzieli na train i val)."
        ),
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proporcje podziału train/val/test w trybie 'split' (domyślnie: 0.70 0.15 0.15)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporcja walidacji przy --test-source existing (domyślnie: 0.15)",
    )
    parser.add_argument(
        "--strategy",
        choices=["scene", "random"],
        default="scene",
        help="Strategia podziału: 'scene' (zalecana, scene-aware bez przecieku) lub 'random' (losowa)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Usuń wykryte identyczne duplikaty klatek (domyślnie: włączone)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_false",
        dest="deduplicate",
        help="Wyłącz usuwanie zduplikowanych klatek",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=["rgb", "pair"],
        default="rgb",
        help="Kryterium duplikatu: 'rgb' (identyczny obraz RGB) lub 'pair' (identyczny RGB oraz głębia)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości dla powtarzalności (domyślnie: 42)",
    )
    parser.add_argument(
        "--add-header",
        action="store_true",
        default=False,
        help="Dodaj nagłówek 'image,depth' w plikach CSV (domyślnie: False)",
    )

    args = parser.parse_args()

    project_root = Path.cwd()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else project_root / args.dataset_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    print("=" * 70)
    print(" Przygotowanie i audyt datasetu do sieci neuronowej")
    print(f" Katalog datasetu: {dataset_dir}")
    print(f" Katalog wyjściowy: {output_dir}")
    print(f" Strategia podziału: {args.strategy.upper()} | Źródło testu: {args.test_source}")
    print("=" * 70)

    # 1. Wczytanie danych
    print("\n[1/4] Wczytywanie i sprawdzanie integralności plików...")
    train_pairs, test_pairs = load_dataset_pairs(dataset_dir)
    print(f"  Znaleziono par w nyu2_train: {len(train_pairs):,}")
    print(f"  Znaleziono par w nyu2_test:  {len(test_pairs):,}")

    if not train_pairs:
        print("BŁĄD: Nie znaleziono danych treningowych w katalogu:", dataset_dir)
        sys.exit(1)

    train_valid, train_errors, train_sizes = verify_file_integrity(train_pairs)
    test_valid, test_errors, test_sizes = verify_file_integrity(test_pairs)

    all_sizes = {**train_sizes, **test_sizes}

    if train_errors or test_errors:
        print(f"  Znaleziono błędy integralności: {len(train_errors) + len(test_errors)}")
        for err in (train_errors + test_errors)[:5]:
            print("   -", err)
    else:
        print("  Wszystkie pliki istnieją, są czytelne i niepuste (OK).")

    # 2. Weryfikacja powtórzeń (duplikatów)
    print("\n[2/4] Weryfikacja powtórzeń (duplikatów) w zbiorze danych...")
    rgb_dupes, depth_dupes, pair_dupes, rgb_hashes, depth_hashes = find_duplicates(
        train_valid, cached_sizes=all_sizes
    )

    num_rgb_dupes = sum(len(paths) - 1 for paths in rgb_dupes.values())
    num_depth_dupes = sum(len(paths) - 1 for paths in depth_dupes.values())
    num_pair_dupes = sum(len(paths) - 1 for paths in pair_dupes.values())

    print(f"  Zidentyfikowane grupy identycznych obrazów RGB: {len(rgb_dupes)} (nadmiarowych klatek: {num_rgb_dupes})")
    for h, paths in list(rgb_dupes.items())[:5]:
        sample_names = [p.name for p in paths[:4]]
        scene = paths[0].parent.name
        print(f"    * Scena '{scene}': {len(paths)} identycznych klatek RGB (np. {sample_names})")

    print(f"  Zidentyfikowane grupy identycznych map głębokości: {len(depth_dupes)}")
    print(f"  Zidentyfikowane identyczne pary (RGB + Depth): {len(pair_dupes)}")

    # Sprawdzenie przecieku między nyu2_train a nyu2_test
    leakage_count = 0
    if test_valid:
        test_rgb_sizes = {test_sizes[tp]: tp for tp, _, _ in test_valid}
        test_rgb_hashes = {compute_file_hash(p): p for p, _, _ in test_valid}
        for rgb_p, _, _ in train_valid:
            if train_sizes[rgb_p] in test_rgb_sizes:
                h = compute_file_hash(rgb_p)
                if h in test_rgb_hashes:
                    leakage_count += 1
        print(f"  Wykryte kolizje (przeciek) między nyu2_train a nyu2_test: {leakage_count}")

    # Deduplikacja
    removed_pairs: List[Tuple[Path, Path, str]] = []
    if args.deduplicate and (num_rgb_dupes > 0 or num_pair_dupes > 0):
        train_processed, removed_pairs = deduplicate_pairs(
            train_valid, rgb_hashes=rgb_hashes, depth_hashes=depth_hashes, dedup_mode=args.dedup_mode
        )
        print(f"  Zastosowano deduplikację (tryb '{args.dedup_mode}'): usunięto {len(removed_pairs)} powtórzonych klatek.")
        print(f"  Pozostało unikalnych próbek: {len(train_processed):,}")
    else:
        train_processed = train_valid
        print("  Deduplikacja pominięta lub brak próbek do usunięcia.")

    # 3. Podział na zbiory
    print("\n[3/4] Podział na zbiór uczący, walidacyjny i testowy...")
    scene_allocations = {}

    if args.test_source == "split":
        # Dzielimy nyu2_train na train, val, test
        if args.strategy == "scene":
            train_set, val_set, test_set, scene_allocations = split_by_scenes(
                train_processed, tuple(args.ratios), seed=args.seed
            )
        else:
            train_set, val_set, test_set = split_random(
                train_processed, tuple(args.ratios), seed=args.seed
            )
    else:
        # Używamy nyu2_test jako test, a nyu2_train dzielimy na train i val
        test_set = test_valid
        val_r = args.val_ratio
        train_r = 1.0 - val_r
        if args.strategy == "scene":
            train_set, val_set, _, scene_allocations = split_by_scenes(
                train_processed, (train_r, val_r, 0.0), seed=args.seed
            )
            scene_allocations["test_scenes"] = ["nyu2_test_official"]
        else:
            train_set, val_set, _ = split_random(
                train_processed, (train_r, val_r, 0.0), seed=args.seed
            )

    total_split = len(train_set) + len(val_set) + len(test_set)
    p_train = len(train_set) / total_split * 100
    p_val = len(val_set) / total_split * 100
    p_test = len(test_set) / total_split * 100

    print(f"  Podział zakończony pomyślnie:")
    print(f"    - Zbiór uczący (train):      {len(train_set):6,} próbek ({p_train:5.2f}%)")
    print(f"    - Zbiór walidacyjny (val):   {len(val_set):6,} próbek ({p_val:5.2f}%)")
    print(f"    - Zbiór testowy (test):      {len(test_set):6,} próbek ({p_test:5.2f}%)")
    print(f"    - Razem:                     {total_split:6,} próbek")

    if scene_allocations:
        print(f"  Rozkład scen (brak przecieku / no data leakage):")
        print(f"    - Liczba scen w train: {len(scene_allocations.get('train_scenes', []))}")
        print(f"    - Liczba scen w val:   {len(scene_allocations.get('val_scenes', []))}")
        print(f"    - Liczba scen w test:  {len(scene_allocations.get('test_scenes', []))}")

    # 4. Zapis wyników
    print(f"\n[4/4] Zapisywanie plików CSV i raportu w '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    report_json = output_dir / "dataset_summary.json"

    save_split_csv(train_csv, train_set, relative_to=project_root, add_header=args.add_header)
    save_split_csv(val_csv, val_set, relative_to=project_root, add_header=args.add_header)
    save_split_csv(test_csv, test_set, relative_to=project_root, add_header=args.add_header)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "strategy": args.strategy,
        "test_source": args.test_source,
        "seed": args.seed,
        "deduplication": {
            "enabled": args.deduplicate,
            "mode": args.dedup_mode,
            "removed_count": len(removed_pairs),
            "rgb_duplicate_groups": len(rgb_dupes),
            "redundant_rgb_frames": num_rgb_dupes,
        },
        "counts": {
            "total_samples": total_split,
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "percentages": {
            "train": round(p_train, 2),
            "val": round(p_val, 2),
            "test": round(p_test, 2),
        },
        "scenes": scene_allocations,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Zapisano: {train_csv}")
    print(f"  Zapisano: {val_csv}")
    print(f"  Zapisano: {test_csv}")
    print(f"  Zapisano: {report_json}")

    print("\n" + "=" * 70)
    print(" Gotowe! Dane zostały przygotowane i są gotowe do treningu.")
    print("=" * 70)


if __name__ == "__main__":
    main()
