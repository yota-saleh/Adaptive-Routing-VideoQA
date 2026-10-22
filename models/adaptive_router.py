"""Compact public sample of adaptive question-conditioned routing.

This is a method illustration, not the complete experiment implementation.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveRouter(nn.Module):
    """Score segments, select an adaptive top-k set, and aggregate evidence."""

    def __init__(self, hidden_size: int = 768, k_min: int = 1, k_max: int = 6):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.complexity = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, segments: torch.Tensor, question: torch.Tensor, tau: torch.Tensor):
        question = question.unsqueeze(1).expand(-1, segments.size(1), -1)
        scores = self.score(torch.cat([segments, question], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores / tau.clamp_min(1e-4), dim=-1)

        # The full implementation uses its complete adaptive-k and
        # straight-through routing path. This sample shows the public idea.
        k_fraction = self.complexity(question[:, 0]).squeeze(-1)
        k = (self.k_min + k_fraction * (self.k_max - self.k_min)).round().long()
        k = k.clamp(self.k_min, self.k_max)
        top_indices = weights.topk(self.k_max, dim=-1).indices
        ranks = torch.arange(self.k_max, device=segments.device).unsqueeze(0)
        mask = ranks < k.unsqueeze(1)
        selected = torch.zeros_like(weights).scatter(1, top_indices, mask.float())
        selected = selected * weights
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = (segments * selected.unsqueeze(-1)).sum(dim=1)
        return evidence, selected, k
