"""Create the public, aggregate comparison figure from reported metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
conditions = [
    "M1 static\naggregation", "M2 uniform\npooling", "M3 fixed\nrouting",
    "Adaptive\ncollapse-prone", "Adaptive\nanti-collapse\n(context)",
    "Adaptive\nanti-collapse\n(no context)", "Question-grounded\nanti-collapse",
]
accuracy = pd.DataFrame(
    {"Top-1 all": [34.81, 46.04, 45.78, 45.86, 46.03, 35.26, 35.32],
     "Top-3 all": [51.67, 63.84, 63.48, 63.53, 64.00, 52.58, 51.91]},
    index=conditions,
)
routing = pd.read_csv(ROOT / "metrics" / "adaptive_conditions.csv")
routing_labels = ["Collapse-prone", "Anti-collapse\n(context)",
                  "Anti-collapse\n(no context)", "Question-grounded\n(no context)"]

fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
x = np.arange(len(accuracy))
width = 0.36
left.bar(x - width / 2, accuracy["Top-1 all"], width, label="Top-1 All", color="#2878B5")
left.bar(x + width / 2, accuracy["Top-3 all"], width, label="Top-3 All", color="#F28E2B")
left.set_title("MSRVTT-QA answer accuracy", weight="bold")
left.set_ylabel("Accuracy (%)")
left.set_xticks(x, accuracy.index, rotation=25, ha="right")
left.set_ylim(0, 75)
left.grid(axis="y", alpha=0.25)
left.legend(frameon=False)

x2 = np.arange(len(routing))
right.plot(x2, routing["mean_entropy"], marker="o", linewidth=2.5, label="Routing entropy")
right.plot(x2, routing["effective_segments"], marker="s", linewidth=2.5, label="Effective segments")
right.plot(x2, routing["max_position_share"] * 4, marker="^", linewidth=2.5,
           label="Max-position share x 4")
right.set_title("Routing diagnostics", weight="bold")
right.set_ylabel("Reported value")
right.set_xticks(x2, routing_labels, rotation=25, ha="right")
right.grid(axis="y", alpha=0.25)
right.legend(frameon=False, fontsize=9)

fig.suptitle("Adaptive Temporal Routing: Accuracy and Routing Behavior", fontsize=16, weight="bold")
fig.savefig(ROOT / "figures" / "adaptive_routing_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
