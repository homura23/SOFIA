from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Common suffixes that appear on vocal stems; used to recover the base song stem
VOCAL_SUFFIXES = ["_vocal_16k", "_vocals_16k", "_vocal", "_vocals", "-vocal", "-vocals"]


def list_audio_files(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS and p.is_file()]


def normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suf in VOCAL_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


@dataclass
class PairedSource:
    label_name: str
    full_dirs: List[Path]
    vocal_dirs: List[Path]
    max_count: int


def build_full_lookup(full_dirs: List[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for d in full_dirs:
        if not d.exists():
            print(f"[WARN] Full-audio dir missing: {d}")
            continue
        for f in list_audio_files(d):
            key = normalize_stem(f)
            if key in lookup:
                # Keep first occurrence to avoid oscillation across dirs
                continue
            lookup[key] = f
    return lookup


def pair_tracks(spec: PairedSource) -> List[Tuple[str, str, str]]:
    lookup = build_full_lookup(spec.full_dirs)
    paired: List[Tuple[str, str, str]] = []
    missing = 0
    total_vocals = 0

    for vdir in spec.vocal_dirs:
        if not vdir.exists():
            print(f"[WARN] Vocal dir missing: {vdir}")
            continue
        vfiles = list_audio_files(vdir)
        total_vocals += len(vfiles)
        random.shuffle(vfiles)
        for vf in vfiles:
            key = normalize_stem(vf)
            full = lookup.get(key)
            if full is None:
                missing += 1
                continue
            paired.append((str(full), str(vf), spec.label_name))
            if len(paired) >= spec.max_count:
                break
        if len(paired) >= spec.max_count:
            break

    print(f"[INFO] {spec.label_name}: matched {len(paired)}/{total_vocals} vocals; missing {missing}")
    return paired


def write_csv(path: Path, rows: List[Tuple[str, str, int, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["full_path", "vocal_path", "label", "source"])
        for full_p, vocal_p, lbl, src in rows:
            writer.writerow([full_p, vocal_p, lbl, src])
    print(f"[OK] wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test CSVs for metric learning with paired full+vocal audio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train_human_udio.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/valid_human_udio.csv"))
    parser.add_argument("--test-suno35-csv", type=Path, default=Path("data/test_suno3.5.csv"))
    parser.add_argument("--test-human2-csv", type=Path, default=Path("data/test_human2.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--exclude-train-sources",
        type=str,
        default="sunov5,ace,acestep",
        help="Comma-separated source names to exclude from training set (case-insensitive)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    preset_train = [
        PairedSource(
            label_name="human",
            full_dirs=[Path("data/datasets/laion-disco/0001_wav")],
            vocal_dirs=[Path("data/datasets/laion-disco/0001_wav_vocal_16k")],
            max_count=15_000,
        ),
        PairedSource(
            label_name="udio",
            full_dirs=[Path("data/datasets/udio_audio")],
            vocal_dirs=[Path("data/datasets/udio_dataset/audio/vocals")],
            max_count=15_000,
        ),
    ]

    preset_test_suno35 = [
        PairedSource(
            label_name="suno3.5",
            full_dirs=[Path("data/datasets/suno_dataset/audio/wav")],
            vocal_dirs=[Path("data/datasets/suno_dataset/audio/wav_vocal_16k")],
            max_count=5_000,
        )
    ]

    preset_test_human2 = [
        PairedSource(
            label_name="human",
            full_dirs=[Path("data/datasets/laion-disco/0002_wav")],
            vocal_dirs=[Path("data/datasets/laion-disco/0002_wav_vocal_16k")],
            max_count=5_000,
        )
    ]

    def label_for(name: str) -> int:
        return 0 if name == "human" else 1

    def gather_rows(specs: List[PairedSource]) -> List[Tuple[str, str, int, str]]:
        rows: List[Tuple[str, str, int, str]] = []
        for spec in specs:
            pairs = pair_tracks(spec)
            lbl = label_for(spec.label_name)
            rows.extend([(full, vocal, lbl, spec.label_name) for full, vocal, _src in pairs])
        return rows

    exclude_train = {s.strip().lower() for s in args.exclude_train_sources.split(',') if s.strip()}

    train_rows_all = [r for r in gather_rows(preset_train) if r[3].lower() not in exclude_train]
    random.shuffle(train_rows_all)

    cut = int(len(train_rows_all) * 0.1)
    final_val = train_rows_all[:cut]
    final_train = train_rows_all[cut:]

    test_rows_suno35 = gather_rows(preset_test_suno35)
    test_rows_human2 = gather_rows(preset_test_human2)
    test_rows = test_rows_suno35 + test_rows_human2
    random.shuffle(final_train)
    random.shuffle(final_val)

    write_csv(args.train_csv, final_train)
    write_csv(args.val_csv, final_val)
    write_csv(args.test_suno35_csv, test_rows_suno35)
    write_csv(args.test_human2_csv, test_rows_human2)
    write_csv(args.test_csv, test_rows)


if __name__ == "__main__":
    main()
