from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch
import torchaudio
import yaml

from sofia.encoders.base import BaseEncoder
from sofia.utils.audio import crop_or_pad, ensure_channels


class RawNetEncoder(BaseEncoder):
    def __init__(
        self,
        config_path: str,
        weights: str,
        repo_path: Optional[str] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(name="rawnet", target_sr=16000, channels=1, embed_dim=0, device=device)
        repo = repo_path or os.path.dirname(config_path)
        model_file = os.path.join(repo, "model.py")
        module_name = "rawnet_local_model"

        # Load RawNet model module explicitly from file to avoid collisions with other 'model.py'
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load RawNet module from {model_file}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_module
        spec.loader.exec_module(model_module)
        RawNet = getattr(model_module, "RawNet")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        d_args = cfg["model"]

        self.model = RawNet(d_args=d_args, device=device).to(device)
        self.target_len = d_args.get("nb_samp")
        embed_dim = d_args.get("nb_fc_node", 1024)
        self.embed_dim = embed_dim

        state = torch.load(weights, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[RawNet] Loaded with missing={len(missing)} unexpected={len(unexpected)} params")

        self.model.eval()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        if sample_rate != self.target_sr:
            audio = torchaudio.functional.resample(audio, sample_rate, self.target_sr)
            sample_rate = self.target_sr

        audio = ensure_channels(audio, self.channels)

        if self.target_len:
            audio = crop_or_pad(audio, self.target_len, random_crop=False)

        audio = audio.squeeze(1)  # RawNet expects [B, T]
        _, emb = self.model(audio, return_embedding=True)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb
