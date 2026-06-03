from __future__ import annotations

import os
from typing import Optional

import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from sofia.encoders.base import BaseEncoder
from sofia.utils.audio import ensure_channels


class Wave2Vec2Encoder(BaseEncoder):
    def __init__(
        self,
        weights: Optional[str] = None,
        device: str | torch.device = "cpu",
        layer_pooling: str = "layer_mean",
    ) -> None:
        model_path = weights or os.getenv("WAV2VEC_WEIGHTS", "weights/wav2vec")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        hidden_size = int(getattr(model.config, "hidden_size", 768))
        target_sr = int(processor.sampling_rate)
        super().__init__(name="wave", target_sr=target_sr, channels=1, embed_dim=hidden_size, device=device)

        self.processor = processor
        self.model = model
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        pooling = layer_pooling.lower().strip()
        if pooling not in {"layer_mean", "last_layer"}:
            raise ValueError(f"Unsupported Wave2Vec2 layer_pooling '{layer_pooling}'. Use 'layer_mean' or 'last_layer'.")
        self.layer_pooling = pooling

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        if sample_rate != self.target_sr:
            audio = torchaudio.functional.resample(audio, sample_rate, self.target_sr)

        audio = ensure_channels(audio, self.channels)
        audio = torch.clamp(audio, -1.0, 1.0)

        feats = []
        model_device = next(self.model.parameters()).device
        for i in range(audio.size(0)):
            wav = audio[i, 0].detach().cpu()
            inputs = self.processor(wav, sampling_rate=self.target_sr, return_tensors="pt").to(model_device)
            out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("Wave2Vec2 forward did not return hidden_states.")

            if self.layer_pooling == "last_layer":
                feat = hs[-1].mean(dim=1).squeeze(0)
            else:
                stacked = torch.stack(hs, dim=0).squeeze(1)  # [L, T, D]
                feat = stacked.mean(dim=1).mean(dim=0)  # [D]

            feats.append(feat)

        return torch.stack(feats, dim=0)
