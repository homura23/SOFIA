from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from sofia.encoders.base import BaseEncoder
from sofia.utils.audio import ensure_channels


def _candidate_repos(repo_path: Optional[str]) -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    code_root = Path(__file__).resolve().parents[2]
    repos = [Path(p).expanduser() for p in (repo_path, os.getenv("MUQ_REPO")) if p]
    repos.extend([project_root / "third_party" / "muq", code_root / "muq", code_root / "MuQ"])
    return repos


def _load_muq(repo_path: Optional[str]):
    import_error: ModuleNotFoundError | None = None
    for repo in _candidate_repos(repo_path):
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from muq import MuQ

            return MuQ
        except ModuleNotFoundError as exc:
            import_error = exc
            continue
    try:
        from muq import MuQ

        return MuQ
    except ModuleNotFoundError as exc:
        import_error = import_error or exc
        raise ModuleNotFoundError(
            "muq is not importable. Install it, set MUQ_REPO, "
            "or set audio_branches.muq.repo_path to the MuQ source root."
        ) from import_error


class MuQEncoder(BaseEncoder):
    def __init__(
        self,
        weights: Optional[str] = None,
        repo_path: Optional[str] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(name="muq", target_sr=24000, channels=1, embed_dim=1024, device=device)
        MuQ = _load_muq(repo_path)
        weight_path = weights or os.getenv("MUQ_WEIGHTS", "weights/muq/muq_weights")
        self.model = MuQ.from_pretrained(weight_path).to(device).eval()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if sample_rate != self.target_sr:
            audio = torchaudio.functional.resample(audio, sample_rate, self.target_sr)
            sample_rate = self.target_sr

        audio = ensure_channels(audio, self.channels)
        out = self.model(audio, output_hidden_states=True)
        h = out.last_hidden_state  # [B, T, D] or [T, D]

        if h.dim() == 2:
            h = h.unsqueeze(0)

        # Mean pooling over time
        h = h.mean(dim=1)  # [B, 1024]
        return h