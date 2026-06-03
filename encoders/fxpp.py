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
    repo_values = [
        repo_path,
        os.getenv("FXPP_REPO"),
        os.getenv("FXENCODER_PLUSPLUS_REPO"),
    ]
    project_root = Path(__file__).resolve().parents[1]
    code_root = Path(__file__).resolve().parents[2]
    repos = [Path(p).expanduser() for p in repo_values if p]
    repos.extend(
        [
            project_root / "third_party" / "fxencoder_plusplus",
            code_root / "fxplusplus" / "fx_code" / "Fx-Encoder_PlusPlus-main",
            code_root / "Fx-Encoder_PlusPlus-main",
        ]
    )
    return repos


def _load_fxencoder(repo_path: Optional[str]):
    import_error: ModuleNotFoundError | None = None
    for repo in _candidate_repos(repo_path):
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from fxencoder_plusplus.model import FxEncoderPlusPlus, get_model_path

            return FxEncoderPlusPlus, get_model_path
        except ModuleNotFoundError as exc:
            import_error = exc
            continue
    try:
        from fxencoder_plusplus.model import FxEncoderPlusPlus, get_model_path

        return FxEncoderPlusPlus, get_model_path
    except ModuleNotFoundError as exc:
        import_error = import_error or exc
        raise ModuleNotFoundError(
            "fxencoder_plusplus is not importable. Install it, set FXPP_REPO, "
            "or set audio_branches.fxpp.repo_path to the Fx-Encoder++ source root."
        ) from import_error


class FxPPEncoder(BaseEncoder):
    def __init__(
        self,
        weights: Optional[str] = None,
        repo_path: Optional[str] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(name="fxpp", target_sr=44100, channels=2, embed_dim=128, device=device)
        FxEncoderPlusPlus, get_model_path = _load_fxencoder(repo_path)
        # Ensure CLAP and extractor modules are disabled per requirement
        model = FxEncoderPlusPlus(
            embed_dim=2048,
            audio_clap_module=False,
            text_clap_module=False,
            extractor_module=False,
            device=str(device),
        )

        ckpt_path = weights if weights else get_model_path(model_name="default")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            # remove possible 'module.' prefix
            if next(iter(sd.keys())).startswith("module"):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False

        self.model = model

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if sample_rate != self.target_sr:
            audio = torchaudio.functional.resample(audio, sample_rate, self.target_sr)
            sample_rate = self.target_sr

        audio = ensure_channels(audio, self.channels)
        audio = torch.clamp(audio, -1.0, 1.0)
        emb = self.model.get_fx_embedding(audio)
        return emb
