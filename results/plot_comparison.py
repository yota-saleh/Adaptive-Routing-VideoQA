"""Draw the aggregate comparison figure from results/metrics/adaptive_conditions.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
routing = pd.read_csv(ROOT / "metrics" / "adaptive_conditions.csv")

# Uniform-pooling reference M2 (from the companion manuscript) and the four adaptive routers.
models = ["M2 uniform\npooling", "Collapse-\nprone", "Anti-collapse\n(context)",
          "Anti-collapse\n(no context)", "Question-grounded\n(no context)"]
top1 = [46.04] + list(100 * routing["top1_all"])
top3 = [63.84] + list(100 * routing["top3_all"])
labels = ["Collapse-\nprone", "Anti-collapse\n(context)", "Anti-collapse\n(no context)",
          "Question-grounded\n(no context)"]

plt.rcParams.update({"font.size": 10})
fig, (left, mid, right) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.45, 1, 1]})

x = np.arange(len(models)); w = 0.38
left.bar(x - w / 2, top1, w, label="Top-1", color="#2E6DB4")
left.bar(x + w / 2, top3, w, label="Top-3", color="#C0571F", hatch="//", edgecolor="white")
left.set_xticks(x, models, fontsize=8.5)
left.set_ylabel("Test accuracy (%)"); left.set_ylim(0, 75); left.legend(frameon=False)
left.set_title("(a) MSRVTT-QA accuracy", loc="left")

x2 = np.arange(len(labels))
mid.bar(x2, routing["effective_segments"], 0.6, color="#2E6DB4")
for i, v in enumerate(routing["effective_segments"]):
    mid.text(i, v + 0.06, f"{v:.2f}", ha="center")
mid.set_xticks(x2, labels, fontsize=8.5); mid.set_ylim(0, 4.3)
mid.set_ylabel("Effective segments, exp(mean entropy)")
mid.set_title("(b) Weight spread", loc="left")

w2 = 0.38
right.bar(x2 - w2 / 2, 100 * routing["top_weight_share"], w2, label="Top-weight share", color="#2E6DB4")
right.bar(x2 + w2 / 2, 100 * routing["mean_confidence"], w2, label="Mean confidence", color="#5E8C31",
          hatch="..", edgecolor="white")
right.set_xticks(x2, labels, fontsize=8.5); right.set_ylim(0, 110)
right.set_ylabel("Percent")
right.set_title("(c) Concentration and confidence", loc="left"); right.legend(frameon=False)

for a in (left, mid, right):
    a.spines[["top", "right"]].set_visible(False)

fig.savefig(ROOT / "figures" / "adaptive_routing_comparison.png", dpi=300)
plt.close(fig)
