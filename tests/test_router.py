"""Shape and gradient-path checks for the released router and objectives."""

import torch

from models import AdaptiveQuestionGuidedRouter
from training.objectives import (entropy_floor_loss, first_segment_penalty, load_balance_loss,
                                 routing_entropy, segment_ranking_loss, selected_mean,
                                 usage_diversity_loss)


def test_router_and_gradient_paths():
    torch.manual_seed(0)
    router = AdaptiveQuestionGuidedRouter()
    seg = torch.randn(4, 8, 768, requires_grad=True)
    q = torch.randn(4, 768)
    routed, w, aux = router(seg, q, tau_t=0.7)
    assert routed.shape == (4, 768) and w.shape == (4, 8)
    assert ((aux["k"] >= 1) & (aux["k"] <= 6)).all()
    assert torch.allclose(w.sum(1), torch.ones(4), atol=1e-5)

    # Entropy terms reach the scorer through the straight-through weights.
    router.zero_grad()
    (routing_entropy(w) + entropy_floor_loss(w)).backward(retain_graph=True)
    assert router.score_proj.weight.grad.abs().sum() > 0

    # Mask-based terms and the detached ranking loss carry no gradient.
    mask = aux["selected_mask"]
    for term in (load_balance_loss(mask), first_segment_penalty(mask), usage_diversity_loss(mask),
                 segment_ranking_loss(aux["scores"], mask)):
        assert not term.requires_grad

    # The selected mean reaches the segment features but not the scorer.
    router.zero_grad()
    selected_mean(seg, mask).sum().backward()
    assert router.score_proj.weight.grad is None or router.score_proj.weight.grad.abs().sum() == 0
