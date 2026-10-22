"""Small public sample showing optional auxiliary-context fusion."""

from __future__ import annotations

import torch
from torch import nn


class ContextFusion(nn.Module):
    """Fuse routed video evidence, question, and optional context vectors."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, video_evidence, question_cls, context_cls=None):
        if context_cls is None:
            context_cls = torch.zeros_like(question_cls)
        return self.projection(torch.cat([video_evidence, question_cls, context_cls], dim=-1))
