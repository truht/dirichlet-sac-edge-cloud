#!/usr/bin/env python3
"""Regenerate fig_pareto_scatter.pdf from Table 3's verified numbers
(main.tex tab:baseline, synced 2026-10-15/16: 1200 GPU_LOAD_BASED cloud-share
corrected 0.0->2.2, 2000 row added). x = cloud-offload share (%),
y = drops per episode (log scale, floor-clipped for true zeros),
color = policy, marker size = device count. SAC in red per the existing caption."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

device_counts = [300, 600, 900, 1200, 2000]

# (cloud_share_pct, drops) per device count, straight from the corrected Table 3.
data = {
    "Proposed":       [(6.0, 0.007), (6.1, 10.2), (6.2, 329.1), (6.2, 2083.6), (6.1, 25647.9)],
    "GPU_LOAD_BASED":  [(0.0, 0.093), (0.0, 65.7), (0.0, 1000.8), (2.2, 5758.7), (19.9, 35729.2)],
    "LATENCY_BASED":   [(1.6, 0.033), (34.4, 15.7), (55.9, 238.4), (66.3, 1280.5), (77.8, 8172.0)],
    "COMBINED":        [(1.6, 0.037), (34.4, 15.1), (55.9, 234.5), (66.3, 1274.3), (77.8, 8174.8)],
    "RANDOM":          [(50.0, 0.000), (50.0, 0.1), (50.0, 4.5), (50.0, 60.8), (50.0, 2301.6)],
}

colors = {
    "Proposed": "#d62728",       # red, per existing caption
    "GPU_LOAD_BASED": "#1f77b4", # blue
    "LATENCY_BASED": "#2ca02c",  # green
    "COMBINED": "#9467bd",       # purple
    "RANDOM": "#ff7f0e",         # orange
}
markers = {
    "Proposed": "o", "GPU_LOAD_BASED": "s", "LATENCY_BASED": "^",
    "COMBINED": "D", "RANDOM": "v",
}

# marker size scales with device count (as stated in the caption)
size_lo, size_hi = 35, 260
dc_min, dc_max = min(device_counts), max(device_counts)
def size_for(dc):
    frac = (dc - dc_min) / (dc_max - dc_min)
    return size_lo + frac * (size_hi - size_lo)

FLOOR = 0.02  # log-scale floor for true-zero drop values (RANDOM@300, visually distinct, annotated)

fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)

for policy, points in data.items():
    xs = [p[0] for p in points]
    ys = [max(p[1], FLOOR) for p in points]
    sizes = [size_for(dc) for dc in device_counts]
    ax.scatter(xs, ys, s=sizes, c=colors[policy], marker=markers[policy],
               label=policy, alpha=0.85,
               edgecolors="black", linewidths=0.4, zorder=3)
    # connect same-policy points across scale with a thin guide line
    order = np.argsort(xs) if len(set(xs)) > 1 else range(len(xs))
    ax.plot([xs[i] for i in range(len(xs))], [ys[i] for i in range(len(xs))],
            color=colors[policy], alpha=0.25, linewidth=0.8, zorder=1)

ax.set_yscale("log")
ax.set_xlabel("Cloud-offload share (%)", fontsize=9)
ax.set_ylabel("Drops per episode (log scale)", fontsize=9)
ax.set_xlim(-3, 82)
ax.tick_params(labelsize=8)
ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)

legend = ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.02),
                    framealpha=1.0, ncol=1, handletextpad=0.4, borderpad=0.6,
                    labelspacing=0.6, borderaxespad=0)
ax.text(0.98, 0.03,
        "marker size $\\propto$ device count\n(300–2000); RANDOM@300\ndrops=0.000, floor-clipped",
        transform=ax.transAxes, fontsize=5.3, ha="right", va="bottom",
        style="italic", alpha=0.75)

plt.tight_layout(pad=0.6)
plt.savefig("fig_pareto_scatter.pdf", bbox_inches="tight")
print("wrote fig_pareto_scatter.pdf")
