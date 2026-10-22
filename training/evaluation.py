"""Minimal public evaluation helper for video question answering."""

from __future__ import annotations

import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    predictions = logits.topk(k, dim=-1).indices
    correct = predictions.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())
