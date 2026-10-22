"""Compact sample for routing over question-grounded video segments."""

from __future__ import annotations

import torch
from torch import nn

from .adaptive_router import AdaptiveRouter


class GroundedAdaptiveRouter(nn.Module):
    """Apply question-to-video grounding before adaptive routing."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.grounding = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.router = AdaptiveRouter(hidden_size)

    def forward(self, segments, question_tokens, tau):
        grounded, _ = self.grounding(question_tokens, segments, segments)
        grounded_segments = segments + grounded.mean(dim=1, keepdim=True)
        question_cls = question_tokens[:, 0]
        return self.router(grounded_segments, question_cls, tau)
