"""Explanatory (synthetic) diagrams for the beginner primer in the briefing.
These use a small made-up road network, NOT the dataset.
Usage: python concept_figures.py out
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1]
FIG = f"{OUT}/fig"
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREEN, RED, VIOLET = "#2a78d6", "#eb6834", "#008300", "#e34948", "#6250d6"
INK, INK2, MUTED, NET, GRID = "#0b0b0b", "#52514e", "#8a8984", "#9d9c96", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "text.color": INK})

# ---------------- the toy network (metres) ----------------
nodes = np.array([[0, 100], [130, 100], [260, 100], [390, 100], [130, 230], [260, 230], [260, -30]], float)
# arcs as (from, to); two-way roads appear twice
arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),   # main street, two-way
        (1, 4), (4, 1),                                   # side street, two-way
        (2, 5),                                           # one-way street (only 2 -> 5)
        (4, 5), (5, 4),                                   # top street, two-way
        (2, 6), (6, 2)]                                   # south street, two-way
route = [0, 2, 8]  # arc ids: 0->1, 1->2, 2->5
rng = np.random.default_rng(7)


def route_points(step=22):
    pts = []
    for a in route:
        p, q = nodes[arcs[a][0]], nodes[arcs[a][1]]
        L = np.hypot(*(q - p)); n = int(L // step)
        for k in range(n):
            pts.append(p + (q - p) * (k + 0.5) / n)
    return np.array(pts)


true_pts = route_points()
gps = true_pts + rng.normal(0, 7, true_pts.shape)
gps[10] += [16, -6]  # the first fix after the turn drifts east, onto the straight-on street


def draw_arc(ax, a, color=NET, lw=1.4, off=5.0, z=2, label=False):
    i, j = arcs[a]
    p, q = nodes[i], nodes[j]
    d = (q - p) / np.hypot(*(q - p)); n = np.array([-d[1], d[0]])
    twin = (j, i) in arcs
    o = -n * off if twin else 0 * n  # offset to the right of travel direction for two-way pairs
    s, e = p + o + d * 14, q + o - d * 14
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, zorder=z))
    if label:
        m = (s + e) / 2 - n * (10 if twin else -9)
        ax.text(*m, f"arc {a}", fontsize=7, ha="center", va="center", color=INK2, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


# ================= figure C1: nodes, arcs, fixes, route =================
fig, ax = plt.subplots(figsize=(10, 5.6))
for a in range(len(arcs)):
    draw_arc(ax, a, label=True)
for a in route:
    i, j = arcs[a]
    ax.plot(*nodes[[i, j]].T, color=GREEN, lw=9, alpha=0.25, solid_capstyle="round", zorder=1)
for k, (x, y) in enumerate(nodes):
    ax.add_patch(Circle((x, y), 9, fc="white", ec=INK, lw=1.2, zorder=4))
    ax.text(x, y, str(k), ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
ax.scatter(*gps.T, s=26, color=BLUE, zorder=7, edgecolor="white", linewidth=0.7)
for k, off in ((0, (-6, 9)), (len(gps) - 1, (9, -6))):
    ax.annotate(f"t = {k + 1} s", gps[k], xytext=off, textcoords="offset points", fontsize=7.5, color=BLUE,
                fontweight="bold")
c = gps[10]
ax.add_patch(Circle(c, 45, fill=False, ec=ORANGE, lw=1.4, ls="--", zorder=3))
ann = lambda txt, xy, xyt, col=INK: ax.annotate(txt, xy, xytext=xyt, fontsize=8, color=col,  # noqa: E731
                                                arrowprops=dict(arrowstyle="-", color=col if col != INK else MUTED, lw=0.8))
ann("search circle around one GPS fix:\nevery arc touching it is a candidate", c + [30, -34], (335, -30), ORANGE)
ann("node = a point in the road map\n(junction, dead end, or bend)", nodes[4] + [-10, 0], (-65, 222))
ann("green band = the true route\n= arcs 0 → 2 → 8", (60, 104), (-65, 160), GREEN)
ann("two-way street = two arcs,\none per direction (arc 2 and arc 3)", (195, 92), (90, -15))
ann("blue dots = GPS fixes\n(one position per second, noisy)", gps[1] + [0, -3], (-65, 35), BLUE)
ann("one-way street = a single arc\n(arc 8: you may only drive 2 → 5)", (262, 172), (290, 140))
ax.set_xlim(-70, 480); ax.set_ylim(-75, 255); ax.set_aspect("equal"); ax.axis("off")
plt.savefig(f"{FIG}/c1_concepts.png"); plt.close()

# ================= figure C2: how the four files link =================
fig, ax = plt.subplots(figsize=(11, 4.1))
ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.3); ax.axis("off")


def filebox(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=color, lw=1.4))
    ax.text(x + 0.1, y + h - 0.2, title, fontsize=9.5, fontweight="bold", color=color, va="top")
    for k, (l, hl) in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.6 - k * 0.27, l, fontsize=7.6, family="DejaVu Sans Mono", va="top",
                color=INK if hl else MUTED, fontweight="bold" if hl else "normal")


filebox(0.1, 0.25, 2.0, 3.9, "trip.route", [("row  arc_id", False), ("0    0", True), ("1    2", True),
                                              ("2    8", True), ("", False), ("ordered list of", False),
                                              ("arcs driven", False)], GREEN)
filebox(3.1, 0.25, 2.3, 3.9, "trip.arcs", [("row  from  to", False), ("0    0     1", True), ("1    1     0", False),
                                            ("2    1     2", True), ("3    2     1", False), ("…", False),
                                            ("8    2     5", True), ("…", False)], VIOLET)
filebox(6.4, 0.25, 2.2, 3.9, "trip.nodes", [("row  lon      lat", False), ("0    8.5810  49.85", True),
                                             ("1    8.5828  49.85", True), ("2    8.5846  49.85", True),
                                             ("…", False), ("5    8.5846  49.86", True), ("…", False)], INK2)
filebox(9.6, 0.25, 1.9, 3.9, "trip.track", [("lon     lat    t", False), ("8.5811 49.85 1", True),
                                             ("8.5813 49.85 2", True), ("8.5816 49.85 3", True), ("…", False),
                                             ("no IDs at all:", False), ("only position", False),
                                             ("and time", False)], BLUE)
for (x0, x1, y, txt) in [(2.15, 3.05, 3.0, "arc_id is a\nrow number\nin .arcs"),
                         (5.45, 6.35, 3.0, "from / to are\nrow numbers\nin .nodes")]:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.3))
    ax.text((x0 + x1) / 2, y - 0.2, txt, ha="center", va="top", fontsize=6.9, color=INK2)
ax.add_patch(FancyArrowPatch((8.65, 1.6), (9.55, 1.6), arrowstyle="<|-|>", mutation_scale=10, color=RED, lw=1.2, ls="--"))
ax.text(9.1, 1.4, "linked only\nby location\nand order\n(this is the\nwhole task)", ha="center", va="top",
        fontsize=6.9, color=RED)
plt.savefig(f"{FIG}/c2_file_links.png"); plt.close()

# ================= figure C3: HMM / Viterbi lattice =================
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.set_xlim(-0.4, 6.0); ax.set_ylim(-0.7, 3.5); ax.axis("off")
cols = ["fix t=9", "fix t=10", "fix t=11", "fix t=12", "fix t=13"]
cands = [[("arc 2", .80), ("arc 3", .20)],
         [("arc 2", .70), ("arc 3", .20), ("arc 4", .10)],
         [("arc 4", .50), ("arc 8", .42), ("arc 11", .08)],
         [("arc 8", .80), ("arc 4", .20)],
         [("arc 8", .90), ("arc 10", .10)]]
best = ["arc 2", "arc 2", "arc 8", "arc 8", "arc 8"]
pos = {}
for c, (name, cl) in enumerate(zip(cols, cands)):
    x = c * 1.25
    ax.text(x, 3.25, name, ha="center", fontsize=9, fontweight="bold")
    for r, (arc, p) in enumerate(cl):
        y = 2.6 - r * 0.95
        pos[(c, arc)] = (x, y)
for c in range(len(cols) - 1):
    for a, _ in cands[c]:
        for b, _ in cands[c + 1]:
            (x0, y0), (x1, y1) = pos[(c, a)], pos[(c + 1, b)]
            on = best[c] == a and best[c + 1] == b
            ax.plot([x0 + 0.38, x1 - 0.38], [y0, y1], color=GREEN if on else GRID, lw=3 if on else 0.9,
                    zorder=1 if not on else 2)
for c, cl in enumerate(cands):
    top = max(cl, key=lambda t: t[1])[0]
    for arc, p in cl:
        x, y = pos[(c, arc)]
        on = best[c] == arc
        ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.24), 0.76, 0.48, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc="#e3f1e3" if on else "white", ec=GREEN if on else MUTED, lw=1.5 if on else 0.8, zorder=3))
        ax.text(x, y + 0.07, arc, ha="center", va="center", fontsize=8.5, zorder=4, fontweight="bold" if on else "normal")
        ax.text(x, y - 0.13, f"P = {p:.2f}" + ("  ★" if arc == top else ""), ha="center", va="center", fontsize=7,
                color=ORANGE if (arc == top and not on) else INK2, zorder=4)
ax.text(-0.4, -0.45, "Each column is one GPS fix; each box is a candidate arc with the classifier's probability P "
        "(emission). ★ = the classifier's own top pick.\nLines are possible moves between consecutive fixes "
        "(transitions). Viterbi picks the green chain with the best combined score. At t=11 the classifier alone\n"
        "prefers arc 4 (carry straight on east), but the next fixes clearly lie on arc 8 (north). Going east and then "
        "back would need a U-turn, so the decoder chooses arc 8.", fontsize=7.8, color=INK2, va="top")
plt.savefig(f"{FIG}/c3_hmm.png"); plt.close()
print("concept figures written")
