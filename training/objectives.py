"""Routing objectives used in the experiments (Section 3.4 and Table 1).

The selection mask is boolean and the scores passed to the ranking loss are
detached, so the ranking, load-balancing, first-segment, and usage-diversity
terms carry no gradient. The two entropy terms act on the straight-through
weights and therefore reach the scorer and the complexity estimator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats) of the routing weights; minimised with weight 0.02
    in the collapse-prone configuration."""
    w = weights.float()
    return (-(w * w.clamp_min(EPS).log()).sum(dim=1)).mean()


def entropy_floor_loss(weights: torch.Tensor, floor: float = 0.35) -> torch.Tensor:
    """Hinge penalty on entropy below the floor (anti-collapse, weight 0.08)."""
    w = weights.float()
    ent = -(w * w.clamp_min(EPS).log()).sum(dim=1)
    return F.relu(floor - ent).mean()


def load_balance_loss(selected_mask: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation of segment usage (weight 0.01 or 0.03).
    Computed from the boolean mask: no gradient."""
    usage = selected_mask.float().mean(dim=0)
    return (usage.std(unbiased=False) / usage.mean().clamp_min(EPS)) ** 2


def first_segment_penalty(selected_mask: torch.Tensor) -> torch.Tensor:
    """Rate at which segment 1 is selected (anti-collapse, weight 0.05). No gradient."""
    return selected_mask.float()[:, 0].mean()


def usage_diversity_loss(selected_mask: torch.Tensor) -> torch.Tensor:
    """Normalised squared deviation of segment usage from its mean
    (anti-collapse, weight 0.05). No gradient."""
    usage = selected_mask.float().mean(dim=0)
    target = torch.full_like(usage, usage.mean())
    return F.mse_loss(usage, target) / target.mean().clamp_min(EPS)


def segment_ranking_loss(scores_detached: torch.Tensor, selected_mask: torch.Tensor,
                         margin: float = 0.2) -> torch.Tensor:
    """Every selected segment should outscore every unselected one by the margin
    (weight 0.1). The scores are detached in the experiments: no gradient."""
    sel_min = scores_detached.masked_fill(~selected_mask, 1e4).min(dim=1).values
    uns_max = scores_detached.masked_fill(selected_mask, -1e4).max(dim=1).values
    has_both = selected_mask.any(1) & (~selected_mask).any(1)
    if not has_both.any():
        return scores_detached.sum() * 0.0
    return F.relu(margin - (sel_min - uns_max))[has_both].mean()


def info_nce_masked(anchor: torch.Tensor, positive: torch.Tensor, labels: torch.Tensor,
                    video_ids: list[str], temp: float = 0.07) -> torch.Tensor:
    """InfoNCE between the selected-segment mean and q or c (weights 0.2 and 0.1).
    Same-answer and same-video pairs are removed from the negatives."""
    a = F.normalize(anchor.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    sim = a @ p.t() / temp
    B = sim.size(0)
    same_ans = labels.unsqueeze(0) == labels.unsqueeze(1)
    index = {v: i for i, v in enumerate(sorted(set(video_ids)))}
    vid = torch.tensor([index[v] for v in video_ids], device=sim.device)
    same_vid = vid.unsqueeze(0) == vid.unsqueeze(1)
    collide = (same_ans | same_vid) & ~torch.eye(B, dtype=torch.bool, device=sim.device)
    return F.cross_entropy(sim.masked_fill(collide, -1e4), torch.arange(B, device=sim.device))


def selected_mean(segments: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    """Unweighted mean of the selected segments, formed with the boolean mask."""
    m = selected_mask.to(segments.dtype).unsqueeze(-1)
    return (segments * m).sum(1) / m.sum(1).clamp_min(1e-6)
