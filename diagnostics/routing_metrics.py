"""Routing diagnostics of Section 4.4, computed from per-question records.

Each record holds the forward-pass weights (length S), the selected segment
indices, the confidence kappa, and the video identifier.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


def entropy(weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 1e-8, None)
    return -(weights * np.log(w)).sum(axis=1)


def summarize(weights: np.ndarray, selected: list[list[int]], kappa: np.ndarray,
              video_ids: list[str], num_segments: int = 8) -> dict[str, float]:
    """Concentration, position, and question-sensitivity measures (Table 4)."""
    h = entropy(weights)
    top = weights.argmax(axis=1)
    top_share = np.bincount(top, minlength=num_segments) / len(top)
    coverage = [(max(s) - min(s) + 1) / num_segments for s in selected]

    by_video: dict[str, list[frozenset]] = defaultdict(list)
    for v, s in zip(video_ids, selected):
        by_video[v].append(frozenset(s))
    jaccard = [np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])
               for sets in by_video.values() if len(sets) > 1]

    return {
        "mean_k": float(np.mean([len(s) for s in selected])),
        "mean_entropy": float(h.mean()),
        "effective_segments": float(np.exp(h.mean())),
        "mean_max_weight": float(weights.max(axis=1).mean()),
        "most_frequent_top_segment": int(top_share.argmax()) + 1,
        "top_weight_share": float(top_share.max()),
        "first_segment_rate": float(np.mean([0 in s for s in selected])),
        "mean_temporal_coverage": float(np.mean(coverage)),
        "mean_confidence": float(kappa.mean()),
        "mean_fallback_weight": float(1.0 - kappa.mean()),
        "distinct_patterns": len({tuple(sorted(s)) for s in selected}),
        "within_video_jaccard": float(np.mean(jaccard)),
    }


def rank_biserial(values: np.ndarray, correct: np.ndarray) -> float:
    """Rank-biserial correlation between a routing statistic and correctness;
    positive values mean larger values for correct predictions."""
    from scipy.stats import mannwhitneyu
    c = correct.astype(bool)
    u, _ = mannwhitneyu(values[c], values[~c])
    return 2.0 * u / (c.sum() * (~c).sum()) - 1.0
