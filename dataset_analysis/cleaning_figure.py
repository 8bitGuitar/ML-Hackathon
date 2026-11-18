"""Figure for the cleaning results. Usage: python cleaning_figure.py ../data_clean/cleaning_summary.csv out"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

cs = pd.read_csv(sys.argv[1], dtype={"trip": str})
out = sys.argv[2]
tr = cs[cs.partition == "train"].copy()
BLUE, RED, INK2, GRID, MUTED = "#2a78d6", "#e34948", "#52514e", "#e6e5e0", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": MUTED,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
rules = [("C8_jump", "C8 jump"), ("C4_offroad_dwell", "C4 parked off-road"), ("C7_beyond_route", "C7 beyond route"),
         ("C6_far_from_label", "C6 far from label"), ("C4_dwell_thinned", "C4 dwell thinned")]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axs[0]
vals = [tr[k].sum() for k, _ in rules]
b = ax.barh([n for _, n in rules], vals, color=RED, height=0.6)
for bb, v in zip(b, vals):
    ax.text(v, bb.get_y() + bb.get_height() / 2, f" {v:,}", va="center", fontsize=8, color=INK2)
ax.invert_yaxis(); ax.set_xlabel("train fixes removed"); ax.set_title("Fixes removed per rule (train)", loc="left",
                                                                     fontweight="bold")
ax.set_xlim(right=max(vals) * 1.25)
ax = axs[1]
tr["kept"] = tr.kept_for_training / tr.fixes * 100
s = tr.sort_values("kept").reset_index(drop=True)
ax.bar(range(len(s)), s.kept, color=[RED if v < 70 else BLUE for v in s.kept], width=0.8)
ax.set_ylim(0, 100); ax.set_xticks([]); ax.set_xlabel("train trips, sorted"); ax.set_ylabel("% of fixes kept")
ax.set_title("Share of each trip kept for training", loc="left", fontweight="bold")
for i, r in s.head(4).iterrows():
    ax.text(i, r.kept + 2, r.trip[-3:], rotation=90, fontsize=7, ha="center", va="bottom", color=INK2)
plt.savefig(f"{out}/fig/fig16_cleaning.png")
print("ok")
