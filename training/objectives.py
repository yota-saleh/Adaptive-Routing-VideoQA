"""Selected anti-collapse objective terms used in the experiments."""

from __future__ import annotations

import torch


def entropy_floor_loss(weights: torch.Tensor, floor: float = 0.35) -> torch.Tensor:
    """Penalize routing entropy below the requested floor."""
    safe = weights.clamp_min(1e-8)
    entropy = -(safe * safe.log()).sum(dim=-1)
    return torch.relu(floor - entropy).mean()


def position_penalty(weights: torch.Tensor, position: int = 0) -> torch.Tensor:
    """Discourage a fixed shortcut position from dominating every sample."""
    return weights[:, position].mean()


def diversity_loss(weights: torch.Tensor) -> torch.Tensor:
    """Encourage aggregate usage across temporal positions."""
    usage = weights.mean(dim=0)
    target = torch.full_like(usage, 1.0 / usage.numel())
    return torch.mean((usage - target) ** 2)


def anti_collapse_regularization(weights, entropy_weight=0.08, position_weight=0.05, diversity_weight=0.05):
    return (
        entropy_weight * entropy_floor_loss(weights)
        + position_weight * position_penalty(weights)
        + diversity_weight * diversity_loss(weights)
    )
