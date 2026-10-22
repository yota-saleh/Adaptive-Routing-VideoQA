"""Public routing diagnostics for collapse and shortcut behaviour."""

from __future__ import annotations

import torch


def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=-1)


def effective_segments(weights: torch.Tensor) -> torch.Tensor:
    return routing_entropy(weights).mean().exp()


def positional_usage(weights: torch.Tensor) -> torch.Tensor:
    return weights.mean(dim=0)


def summarize_routing(weights: torch.Tensor) -> dict[str, float]:
    entropy = routing_entropy(weights)
    usage = positional_usage(weights)
    return {
        "mean_entropy": float(entropy.mean().item()),
        "effective_segments": float(entropy.mean().exp().item()),
        "max_position_share": float(usage.max().item()),
        "mean_temporal_coverage": float((weights > 0).float().mean(dim=1).mean().item()),
    }
