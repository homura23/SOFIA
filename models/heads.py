from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 32.0, margin: float = 0.3, easy_margin: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings, dim=1, eps=1e-6), F.normalize(self.weight, dim=1, eps=1e-6))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return logits


class SoftmaxHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        return self.fc(embeddings)
