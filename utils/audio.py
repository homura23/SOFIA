import random
from typing import Tuple

import torch
import torchaudio
import warnings

# Suppress torchaudio TorchCodec migration warnings; avoid deprecated backend setter
warnings.filterwarnings(
    "ignore",
    message=r".*load_with_torchcodec.*",
    category=UserWarning,
    module=r"torchaudio\._backend\.utils",
)


def load_audio(path: str, target_sr: int | None = None, mono: bool = True, normalize: bool = True) -> Tuple[torch.Tensor, int]:
    """Load an audio file and optionally resample/mono it.

    Falls back gracefully if the file cannot be decoded.
    """
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        # Fallback: return empty tensor and target_sr (caller should handle)
        sr = target_sr if target_sr is not None else 44100
        waveform = torch.zeros(1, 1, dtype=torch.float32)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sr is not None and sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    if normalize:
        peak = waveform.abs().max().clamp(min=1e-9)
        waveform = waveform / peak

    return waveform, sr


def separate_vocals(
    waveform: torch.Tensor,
    sample_rate: int,
    model_name: str = "htdemucs",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Separate vocals from a waveform using demucs (no caching).

    Returns a [C, T] tensor at the original sample_rate.
    """
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:
        raise RuntimeError("demucs is required for vocal separation") from exc

    model = get_model(model_name)
    model.to(device)
    model.eval()

    if waveform.dim() == 2:
        wav = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        wav = waveform
    else:
        raise ValueError(f"Unsupported waveform dims: {waveform.dim()}")

    model_sr = int(getattr(model, "samplerate", sample_rate))
    if sample_rate != model_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, model_sr)

    with torch.no_grad():
        # demucs expects [B, C, T]
        stems = apply_model(model, wav.to(device), split=True, overlap=0.25, progress=False)

    sources = getattr(model, "sources", None) or []
    if "vocals" not in sources:
        raise RuntimeError("demucs model does not provide vocals stem")

    vocals_index = sources.index("vocals")
    vocals = stems[:, vocals_index]  # [B, C, T]
    vocals = vocals[0].detach().cpu()

    if model_sr != sample_rate:
        vocals = torchaudio.functional.resample(vocals, model_sr, sample_rate)

    return vocals


def crop_or_pad(waveform: torch.Tensor, target_length: int, random_crop: bool = False) -> torch.Tensor:
    """Crop or zero-pad waveform to target_length samples."""
    current = waveform.shape[-1]
    if current == target_length:
        return waveform

    if current > target_length:
        if random_crop:
            start = random.randint(0, current - target_length)
        else:
            start = max((current - target_length) // 2, 0)
        end = start + target_length
        return waveform[..., start:end]

    pad_total = target_length - current
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return torch.nn.functional.pad(waveform, (pad_before, pad_after))


def ensure_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    """Convert waveform to requested channel count (supports [C, T] or [B, C, T])."""
    if waveform.dim() == 2:
        # [C, T]
        if waveform.shape[0] == channels:
            return waveform
        if channels == 1:
            return waveform.mean(dim=0, keepdim=True)
        if channels == 2:
            if waveform.shape[0] == 1:
                return waveform.repeat(2, 1)
            return waveform[:2, :]
        raise ValueError(f"Unsupported channel count: {channels}")

    if waveform.dim() == 3:
        # [B, C, T]
        b, c, t = waveform.shape
        if c == channels:
            return waveform
        if channels == 1:
            return waveform.mean(dim=1, keepdim=True)
        if channels == 2:
            if c == 1:
                return waveform.repeat(1, 2, 1)
            return waveform[:, :2, :]
        raise ValueError(f"Unsupported channel count: {channels}")

    raise ValueError(f"Unsupported waveform dims: {waveform.dim()}")
