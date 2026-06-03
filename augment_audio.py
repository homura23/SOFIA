#!/usr/bin/env python3
"""
Audio augmentation pipeline for CSV rows.

For each row in the input CSV this script can generate:
- Pitch-shifted waveform
- White-noise added waveform
- Low-bitrate codec re-encode

Important: This script preserves the original sample rate and channel count
of the source audio for all augmentations. No forced mono conversion or fixed
resampling is applied. Multi-channel audio is processed per-channel and saved
with the same number of channels and the original sample rate.

Outputs are written to an output root under dedicated subfolders and new path
columns are appended to the CSV. Existing rows are preserved; new columns are
filled with blank strings on failure. Designed to be model-agnostic.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment audio files and update CSV.")
    parser.add_argument("--input_csv", required=True, help="Path to source CSV with full_path (and optional vocal_path, label).")
    parser.add_argument("--output_csv", required=True, help="Destination CSV with appended augmentation columns.")
    parser.add_argument("--output_root", required=True, help="Root directory where augmented audio will be stored.")
    # Kept for backward-compatibility; not used because we preserve original SR.
    parser.add_argument("--sample_rate", type=int, default=None, help="Deprecated/ignored. Original sample rate is preserved.")
    parser.add_argument("--pitch_shift_semitones", type=float, default=2.0, help="Pitch shift in semitones (positive = up, negative = down). Default: +2 semitones.")
    parser.add_argument("--noise_snr_db", type=float, default=20.0, help="White noise signal-to-noise ratio in dB. Smaller is noisier. Default: 20 dB.")
    parser.add_argument("--codec_format", choices=["mp3", "ogg", "opus", "aac"], default="mp3", help="Codec container/format for re-encode. Default: mp3.")
    parser.add_argument("--codec_bitrate", default="64k", help="Audio bitrate for codec re-encode. Default: 64k.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of rows to process. Useful for sanity checks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing augmented files if they already exist.")
    parser.add_argument("--disable_pitch", action="store_true", help="Skip pitch-shift generation.")
    parser.add_argument("--disable_noise", action="store_true", help="Skip noise-added generation.")
    parser.add_argument("--disable_codec", action="store_true", help="Skip codec re-encode generation.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_audio_preserve(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio preserving original sample rate and channel count.

    Returns audio as float32 with shape (n_samples, n_channels).
    """
    data, sr = sf.read(path.as_posix(), always_2d=True, dtype="float32")
    return data, sr


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    ensure_dir(path.parent)
    sf.write(path.as_posix(), audio, sr, subtype="PCM_16")


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + eps))


def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise at target SNR per channel.

    audio: (n_samples, n_channels)
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    n_samples, n_channels = audio.shape
    out = np.empty_like(audio)
    for ch in range(n_channels):
        sig = audio[:, ch]
        sig_rms = rms(sig)
        if sig_rms == 0.0:
            out[:, ch] = sig
            continue
        noise = np.random.randn(n_samples).astype(np.float32)
        noise = noise / max(rms(noise), 1e-6) * sig_rms * 10 ** (-snr_db / 20.0)
        mixed = sig + noise
        out[:, ch] = np.clip(mixed, -1.0, 1.0)
    return out


def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Apply pitch shift per channel, preserving channel count.

    audio: (n_samples, n_channels)
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    n_samples, n_channels = audio.shape
    out_channels = []
    for ch in range(n_channels):
        shifted = librosa.effects.pitch_shift(audio[:, ch], sr=sr, n_steps=semitones)
        out_channels.append(np.clip(shifted.astype(np.float32), -1.0, 1.0))
    out = np.stack(out_channels, axis=1)
    return out


def reencode_with_ffmpeg(src: Path, dst: Path, fmt: str, bitrate: str, overwrite: bool) -> bool:
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return True
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        src.as_posix(),
        "-b:a",
        bitrate,
        dst.as_posix(),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[ffmpeg] failed for {src} -> {dst}: {result.stderr.strip()}")
            return False
        return True
    except FileNotFoundError:
        print("ffmpeg not found. Install ffmpeg and ensure it is on PATH.")
        return False


def derive_dst(path: Path, root: Path, suffix: str, ext: str) -> Path:
    """Place outputs under root/<suffix>/<stem>_<suffix>.<ext>."""
    return root / suffix / f"{path.stem}_{suffix}.{ext}"


def process_row(
    row: Dict[str, str],
    args: argparse.Namespace,
    output_root: Path,
    has_vocal: bool,
) -> Dict[str, str]:
    out_row = dict(row)
    src_path = Path(row["full_path"]).expanduser()
    vocal_path = Path(row["vocal_path"]).expanduser() if has_vocal else None

    # Initialize columns
    out_row.update(
        {
            "full_path_pitchshift": "",
            "vocal_pitchshift": "",
            "full_path_noise": "",
            "vocal_noise": "",
            "full_path_codec": "",
            "vocal_codec": "",
        }
    )

    try:
        audio, src_sr = load_audio_preserve(src_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] load failed for {src_path}: {exc}")
        return out_row

    # Pitch shift
    if not args.disable_pitch:
        try:
            shifted = apply_pitch_shift(audio, src_sr, args.pitch_shift_semitones)
            dst = derive_dst(src_path, output_root, "pitchshift", "wav")
            save_wav(dst, shifted, src_sr)
            out_row["full_path_pitchshift"] = dst.as_posix()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] pitch shift failed for {src_path}: {exc}")

    # White noise
    if not args.disable_noise:
        try:
            noised = add_white_noise(audio, args.noise_snr_db)
            dst = derive_dst(src_path, output_root, "noise", "wav")
            save_wav(dst, noised, src_sr)
            out_row["full_path_noise"] = dst.as_posix()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] noise add failed for {src_path}: {exc}")

    # Codec re-encode (uses original file to avoid compounding artifacts)
    if not args.disable_codec:
        try:
            codec_ext = args.codec_format
            dst = derive_dst(src_path, output_root, "codec", codec_ext)
            if reencode_with_ffmpeg(src_path, dst, args.codec_format, args.codec_bitrate, args.overwrite):
                out_row["full_path_codec"] = dst.as_posix()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] codec re-encode failed for {src_path}: {exc}")

    # Optional vocal processing mirrors full_path
    if has_vocal and vocal_path is not None:
        try:
            vocal_audio, vocal_sr = load_audio_preserve(vocal_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] vocal load failed for {vocal_path}: {exc}")
            return out_row

        if not args.disable_pitch:
            try:
                shifted = apply_pitch_shift(vocal_audio, vocal_sr, args.pitch_shift_semitones)
                dst = derive_dst(vocal_path, output_root, "pitchshift", "wav")
                save_wav(dst, shifted, vocal_sr)
                out_row["vocal_pitchshift"] = dst.as_posix()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] vocal pitch shift failed for {vocal_path}: {exc}")

        if not args.disable_noise:
            try:
                noised = add_white_noise(vocal_audio, args.noise_snr_db)
                dst = derive_dst(vocal_path, output_root, "noise", "wav")
                save_wav(dst, noised, vocal_sr)
                out_row["vocal_noise"] = dst.as_posix()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] vocal noise add failed for {vocal_path}: {exc}")

        if not args.disable_codec:
            try:
                codec_ext = args.codec_format
                dst = derive_dst(vocal_path, output_root, "codec", codec_ext)
                if reencode_with_ffmpeg(vocal_path, dst, args.codec_format, args.codec_bitrate, args.overwrite):
                    out_row["vocal_codec"] = dst.as_posix()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] vocal codec re-encode failed for {vocal_path}: {exc}")

    return out_row


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_root = Path(args.output_root)

    rows = read_rows(input_csv)
    if args.max_rows:
        rows = rows[: args.max_rows]

    required = {"full_path"}
    missing_required = required - set(rows[0].keys() if rows else [])
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    has_vocal = "vocal_path" in (rows[0].keys() if rows else [])

    augmented_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        augmented = process_row(row, args, output_root, has_vocal)
        augmented_rows.append(augmented)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows")

    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "full_path_pitchshift",
        "vocal_pitchshift",
        "full_path_noise",
        "vocal_noise",
        "full_path_codec",
        "vocal_codec",
    ]
    fieldnames = base_fields + [f for f in extra_fields if f not in base_fields]

    write_rows(output_csv, fieldnames, augmented_rows)
    print(f"Wrote augmented CSV to {output_csv}")


if __name__ == "__main__":
    main()
