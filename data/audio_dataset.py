from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import torch
import torchaudio
from torch.utils.data import Dataset

from sofia.utils.audio import crop_or_pad, ensure_channels, load_audio, separate_vocals


class AudioSampleDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        base_sample_rate: int,
        segment_seconds: float,
        random_crop: bool = True,
        mono: bool = True,
        normalize: bool = True,
        demucs_enabled: bool = True,
        demucs_model: str = "htdemucs",
        demucs_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.items = self._read_csv(csv_path)
        self.base_sample_rate = base_sample_rate
        self.segment_seconds = segment_seconds
        self.random_crop = random_crop
        self.mono = mono
        self.normalize = normalize
        self.demucs_enabled = demucs_enabled
        self.demucs_model = demucs_model
        self.demucs_device = demucs_device

    @staticmethod
    def _read_csv(csv_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        out: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "label" not in reader.fieldnames:
                raise ValueError("CSV must have a 'label' column")

            has_full = "full_path" in reader.fieldnames
            has_vocal = "vocal_path" in reader.fieldnames
            has_legacy = "path" in reader.fieldnames
            if not has_full and not has_legacy:
                raise ValueError("CSV must contain either full_path/vocal_path or path columns")

            for row in reader:
                full_path = row.get("full_path") if has_full else row.get("path")
                vocal_path = row.get("vocal_path") if has_vocal else row.get("vocal_path", "")
                if not full_path:
                    raise ValueError("Missing full_path/path value in CSV row")
                out.append(
                    {
                        "full_path": full_path,
                        "vocal_path": vocal_path or None,
                        "label": int(row["label"]),
                        "source": row.get("source", ""),
                    }
                )
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        full_path = item["full_path"]
        vocal_path = item.get("vocal_path")
        label = int(item["label"])

        target_len = int(self.segment_seconds * self.base_sample_rate) if self.segment_seconds and self.segment_seconds > 0 else None

        def _load(path: str) -> torch.Tensor:
            try:
                wav, _ = load_audio(
                    path,
                    target_sr=self.base_sample_rate,
                    mono=self.mono,
                    normalize=self.normalize,
                )
                if target_len and target_len > 0:
                    return crop_or_pad(wav, target_len, random_crop=self.random_crop)
                return wav
            except Exception:
                ch = 1 if self.mono else 2
                zeros_len = target_len if target_len and target_len > 0 else self.base_sample_rate
                return torch.zeros(ch, max(zeros_len, 1), dtype=torch.float32)

        def _separate_vocals(full_wave: torch.Tensor) -> torch.Tensor | None:
            if not self.demucs_enabled:
                return None
            try:
                vocals = separate_vocals(
                    waveform=full_wave,
                    sample_rate=self.base_sample_rate,
                    model_name=self.demucs_model,
                    device=self.demucs_device,
                )
                vocals = ensure_channels(vocals, 1 if self.mono else 2)
                if target_len and target_len > 0:
                    vocals = crop_or_pad(vocals, target_len, random_crop=self.random_crop)
                return vocals
            except Exception:
                return None

        full_wave = _load(full_path)
        vocal_wave = _load(vocal_path) if vocal_path else None
        if vocal_wave is None or vocal_wave.numel() == 0:
            vocal_wave = _separate_vocals(full_wave)

        if vocal_wave is None or vocal_wave.numel() == 0:
            # Fallback to silence with same shape as full waveform
            vocal_wave = torch.zeros_like(full_wave) if full_wave.numel() > 0 else torch.zeros(1, target_len or 1, dtype=torch.float32)

        return {
            "full": full_wave,
            "vocal": vocal_wave,
            "sample_rate": self.base_sample_rate,
            "label": torch.tensor(label, dtype=torch.long),
            "path_full": full_path,
            "path_vocal": vocal_path or "",
            "source": item.get("source", ""),
        }


class MultiBranchCollate:
    def __init__(
        self,
        branch_cfg: Dict[str, Dict],
        segment_seconds: float,
        random_crop: bool,
    ) -> None:
        self.branch_cfg = branch_cfg
        self.segment_seconds = segment_seconds
        self.random_crop = random_crop

    def __call__(self, batch: List[Dict]) -> Dict[str, Dict]:
        labels = torch.stack([item["label"] for item in batch], dim=0)
        outputs: Dict[str, List[torch.Tensor]] = {}

        # Global cap length (60s at 16k) to bound all branches
        max_cap_len = int(60 * 16000)

        for item in batch:
            full_wave = item.get("full")
            vocal_wave = item.get("vocal")
            base_sr = int(item["sample_rate"])
            for name, cfg in self.branch_cfg.items():
                if not cfg.get("enabled", True):
                    continue
                try:
                    target_sr = int(cfg["target_sr"])
                    channels = int(cfg.get("channels", 1))
                    seg_sec = float(cfg.get("segment_seconds", self.segment_seconds))
                    if seg_sec > 0:
                        target_len = int(seg_sec * target_sr)
                    else:
                        target_len = None

                    # Apply global cap for stability
                    if target_len is None:
                        target_len = max_cap_len
                    else:
                        target_len = min(target_len, max_cap_len)

                    # Route vocals to branches that request them
                    use_vocals = cfg.get("use_vocals", False)
                    audio = vocal_wave if use_vocals else full_wave
                    sr = base_sr
                    if audio is None or audio.numel() == 0:
                        audio = torch.zeros(channels, target_len, dtype=torch.float32)
                        sr = target_sr
                    else:
                        if sr != target_sr:
                            audio = torchaudio.functional.resample(audio, sr, target_sr)
                            sr = target_sr
                        audio = ensure_channels(audio, channels)
                        # Crop/pad to target_len (bounded by cap)
                        if target_len and target_len > 0:
                            audio = crop_or_pad(audio, target_len, random_crop=self.random_crop)

                except Exception:
                    # Fallback to silence for this branch/sample on any failure
                    audio = torch.zeros(channels, max_cap_len, dtype=torch.float32)

                outputs.setdefault(name, []).append(audio)
        stacked = {k: torch.stack(v, dim=0) for k, v in outputs.items()}
        return {"audio": stacked, "label": labels}
