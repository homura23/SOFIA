from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, name: str, target_sr: int, channels: int, embed_dim: int, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.name = name
        self.target_sr = target_sr
        self.channels = channels
        self.embed_dim = embed_dim
        if device is not None:
            self.to(device)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.encode(audio, sample_rate)
