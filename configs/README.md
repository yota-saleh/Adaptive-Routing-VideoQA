# Run configurations

Key settings of the four adaptive runs reported in the paper. All runs use MSRVTT-QA, 16 frames, 8 segments, d_model 768, batch size 8, AdamW (lr 2e-5, weight decay 1e-4), 3 warm-up epochs with cosine decay to 1e-6, at most 20 epochs, patience 6, min_delta 0.0005, mixed precision, gradient clipping at 1.0, no label smoothing, and seed 42.

| Setting | Collapse-prone | Anti-collapse | Anti-collapse, no context | Question-grounded, no context |
| --- | --- | --- | --- | --- |
| Segments scored by the router | raw | raw | raw | question-grounded (after CA2) |
| Context | yes | yes | empty string | empty string |
| k range | 1-6 | 1-6 | 1-6 | 1-6 |
| tau range, base tau | [0.3, 1.0], 0.7 | [0.8, 1.8], 1.2 | [0.8, 1.8], 1.2 | [0.8, 1.8], 1.2 |
| tau schedule | 1.0 to 0.2 | 1.0 to 0.7 | 1.0 to 0.7 | 1.0 to 0.7 |
| Entropy minimisation weight | 0.02 | 0 | 0 | 0 |
| Entropy floor (weight) | none | 0.35 nats (0.08) | 0.35 nats (0.08) | 0.35 nats (0.08) |
| InfoNCE weights (question, context) | 0.2, 0.1 | 0.2, 0.1 | 0.2, 0.1 | 0.2, 0.1 |
| Ranking loss weight (margin) | 0.1 (0.2) | 0.1 (0.2) | 0.1 (0.2) | 0.1 (0.2) |
| Load-balancing weight | 0.01 | 0.03 | 0.03 | 0.03 |
| First-segment / diversity weights | none | 0.05 / 0.05 | 0.05 / 0.05 | 0.05 / 0.05 |
| Selected epoch / last epoch | 8 / 15 | 10 / 16 | 7 / 13 | 11 / 17 |

As explained in the paper, the ranking, load-balancing, first-segment, and diversity terms carry no gradient in this implementation.
