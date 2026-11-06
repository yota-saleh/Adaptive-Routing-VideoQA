"""Adaptive question-guided temporal router (Section 3 of the paper).

This module is an excerpt of the router used in the experiments. The video and
text backbones, the cross-attention blocks, the data pipeline, and the training
loop are not included. Equation numbers refer to the paper.
"""

from __future__ import annotations

import torch
from torch import nn


class QuestionComplexityEstimator(nn.Module):
    """Maps the question [CLS] vector to a complexity score c in (0, 1) (Eq. 2)."""

    def __init__(self, d_model: int = 768, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, q_vec: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(q_vec)).squeeze(-1)


class AdaptiveQuestionGuidedRouter(nn.Module):
    """Question-dependent top-K_c selection with straight-through weights and a
    confidence-aware blend towards the segment mean.

    Args:
        d_model: feature size (768).
        num_segments: number of temporal segments S (8).
        k_min, k_max: bounds of the adaptive number of retained segments (1, 6).
        tau_min, tau_max: temperature range before schedule rescaling.
        base_tau: configured base temperature tau_0 (0.7 or 1.2).
        dropout: dropout inside the scorer and the complexity estimator (0.2).
    """

    def __init__(self, d_model: int = 768, num_segments: int = 8, k_min: int = 1,
                 k_max: int = 6, tau_min: float = 0.8, tau_max: float = 1.8,
                 base_tau: float = 1.2, dropout: float = 0.2):
        super().__init__()
        self.k_min, self.k_max = k_min, k_max
        self.tau_min, self.tau_max, self.base_tau = tau_min, tau_max, base_tau
        # Additive scoring (Eq. 1): w^T Drop(LN(tanh(W_v (s_j + p_j) + W_q q))).
        self.video_proj = nn.Linear(d_model, d_model)
        self.question_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.score_proj = nn.Linear(d_model, 1)
        self.pos_emb = nn.Parameter(torch.zeros(num_segments, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.complexity = QuestionComplexityEstimator(d_model, dropout=dropout)

    def forward(self, segments: torch.Tensor, q_cls: torch.Tensor, tau_t: float):
        """
        Args:
            segments: [B, S, d] segment tokens (raw, or question-grounded for the
                question-grounded variant).
            q_cls: [B, d] question [CLS] vector.
            tau_t: current value of the temperature schedule.
        Returns:
            routed: [B, d] blended routed vector r' (Eq. 5).
            weights: [B, S] straight-through routing weights alpha (Eq. 4).
            aux: diagnostics; the selection mask carries no gradient.
        """
        B, S, _ = segments.shape

        # Eq. 2-3: complexity, number of retained segments, temperature.
        c = self.complexity(q_cls.float())
        k_max = min(self.k_max, S)
        k = (self.k_min + torch.round(c * (k_max - self.k_min))).long()  # no gradient
        tau = (self.tau_min + c * (self.tau_max - self.tau_min)) * (tau_t / self.base_tau)
        tau = tau.clamp_min(1e-6).unsqueeze(1)

        with torch.autocast(device_type=segments.device.type, enabled=False):
            # Eq. 1: the positional embedding enters the scores only.
            v = self.video_proj((segments + self.pos_emb[:S]).float())
            q = self.question_proj(q_cls.float()).unsqueeze(1)
            scores = self.score_proj(self.drop(self.norm(torch.tanh(v + q)))).squeeze(-1)

            # Top-K_c mask from the score ranks.
            order = torch.argsort(scores, dim=1, descending=True)
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(S, device=scores.device).expand(B, S))
            mask = ranks < k.unsqueeze(1)

            # Eq. 4: hard weights forward, soft weights backward.
            soft = torch.softmax(scores / tau, dim=1)
            hard = torch.softmax(scores.masked_fill(~mask, -1e4) / tau, dim=1)
            weights = hard.detach() + soft - soft.detach()

        # Eq. 5: confidence = margin between the two largest weights.
        top2 = torch.topk(weights, k=min(2, S), dim=1).values
        kappa = top2[:, 0] - (top2[:, 1] if S > 1 else 0.0)
        routed = torch.einsum("bs,bsd->bd", weights, segments.float())
        mean_all = segments.float().mean(dim=1)
        routed = kappa.unsqueeze(1) * routed + (1.0 - kappa.unsqueeze(1)) * mean_all

        aux = {
            "scores": scores.detach(),
            "selected_mask": mask.detach(),
            "k": k.detach(),
            "tau": tau.squeeze(1).detach(),
            "complexity": c.detach(),
            "confidence": kappa.detach(),
            "weights_soft": soft.detach(),
        }
        return routed, weights, aux
