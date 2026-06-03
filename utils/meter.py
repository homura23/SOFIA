from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def accuracy(logits, targets) -> float:
    import torch

    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum()
    return float(correct / max(targets.numel(), 1))
