# Metric definitions

All values are computed on the 70,181 MSRVTT-QA test questions used in the paper.

| Column | Meaning |
| --- | --- |
| `top1_all`, `top3_all` | Accuracy over all test questions |
| `top1_inv`, `top3_inv` | Accuracy over questions whose answer is in the training vocabulary |
| `mean_entropy_nats` | Mean entropy of the routing weights used in the forward pass. For the collapse-prone router it is below 1e-27 and is stored as 0.0 |
| `effective_segments` | exp(mean entropy) |
| `mean_k` | Mean number of retained segments |
| `top_weight_share` | Share of questions for which the most frequent top-weighted segment receives the largest weight (Table 4 of the paper) |
| `max_selection_rate` | Share of questions in which the most frequently selected segment is part of the selection |
| `first_segment_rate` | Share of questions in which segment 1 is selected |
| `mean_temporal_coverage` | Mean of (last selected index - first selected index + 1) / 8 |
| `mean_confidence` | Mean margin between the two largest routing weights (kappa) |
