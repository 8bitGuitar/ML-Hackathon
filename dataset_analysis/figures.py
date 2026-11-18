"""Generate all EDA figures from out/trip_stats.csv + out/per_point.npz.
Usage: python figures.py "<Student folder>" out
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_trip, project  # noqa: E402

BASE, OUT = sys.argv[1], sys.argv[2]
FIG = f"{OUT}/fig"
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
FLAG = RED
INK, INK2, MUTED, GRID, NET = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#c9c8c2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "text.color": INK,
})

df = pd.read_csv(f"{OUT}/trip_stats.csv", dtype={"trip": str})
df["tid"] = df.trip.str[-3:]
pp = np.load(f"{OUT}/per_point.npz")
pools = np.load(f"{OUT}/pools.npz")
P = lambda t, k: pp[f"{t}__{k}"]
PCOL = {"train": BLUE, "student_test": ORANGE}
ANOM = {"00000033", "00000054"}


def save(name):
    plt.savefig(f"{FIG}/{name}.png")
    plt.close()


def part_legend(ax, loc="best"):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=BLUE, label="train (80)"),
                       Line2D([], [], marker="s", ls="", color=ORANGE, label="student_test (10)")], loc=loc)


# ---------- 1. world map of trip locations ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
for part, g in df.groupby("partition"):
    ax.scatter(g.lon_c, g.lat_c, s=28, color=PCOL[part], edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-60, 81, 20))
ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
ax.set_title("Trip locations (median GPS position of each trip)", loc="left")
part_legend(ax, "lower left")
for name, (x, y) in {"Europe": (5, 66), "Russia": (55, 66), "N. America": (-110, 45), "S. America": (-75, -15),
                     "SE Asia": (110, 2), "Japan": (128, 44), "S. Africa": (20, -38)}.items():
    ax.text(x, y, name, color=INK2, fontsize=8, style="italic")
save("fig01_world")

# ---------- 2. inventory distributions ----------
fig, axs = plt.subplots(2, 2, figsize=(9, 5.6))
spec = [("n_points", "GPS observations per trip", False, None),
        ("duration_s", "Trip duration (minutes, log)", True, 60),
        ("path_len_km", "GPS path length (km)", False, None),
        ("n_arcs", "Road-network arcs per trip (log)", True, None)]
for ax, (col, title, logx, div) in zip(axs.flat, spec):
    v = df[col] / (div or 1)
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 22) if logx else 22
    ax.hist([v[df.partition == "train"], v[df.partition == "student_test"]], bins=bins, stacked=True,
            color=[BLUE, ORANGE], edgecolor="white", linewidth=0.6)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, loc="left"); ax.set_ylabel("trips")
    ax.axvline(v.median(), color=INK, lw=1, ls="--")
    ax.text(v.median(), ax.get_ylim()[1] * 0.92, f" median {v.median():,.0f}", fontsize=8, color=INK)
part_legend(axs[0, 0], "upper right")
fig.tight_layout()
save("fig02_inventory")

# ---------- 3. sampling & gaps ----------
dt = pools["dt"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1.4, 1.4]})
cats = [("1 s", (dt > 0.5) & (dt <= 1.5)), ("2 s", (dt > 1.5) & (dt <= 2.5)), ("3-10 s", (dt > 2.5) & (dt <= 10)),
        ("11-60 s", (dt > 10) & (dt <= 60)), ("1-10 min", (dt > 60) & (dt <= 600)), (">10 min", dt > 600)]
cnt = [m.sum() for _, m in cats]
ax = axs[0]
bars = ax.bar([c for c, _ in cats], cnt, color=[BLUE] + [FLAG] * 5, width=0.7)
ax.set_yscale("log"); ax.set_title("Time between consecutive fixes", loc="left"); ax.set_ylabel("number of steps (log)")
for b, c in zip(bars, cnt):
    ax.text(b.get_x() + b.get_width() / 2, c * 1.15, f"{c:,}", ha="center", fontsize=7, color=INK2)
ax.tick_params(axis="x", rotation=35)
ax = axs[1]
s = df.sort_values("dt_frac_1s")
ax.bar(range(len(s)), s.dt_frac_1s * 100, color=[FLAG if v < 0.9 else BLUE for v in s.dt_frac_1s], width=0.8)
ax.set_title("Share of 1-second steps per trip", loc="left"); ax.set_ylabel("% of steps"); ax.set_xticks([])
ax.set_xlabel("trips, sorted"); ax.axhline(90, color=INK, lw=0.8, ls="--")
ax.text(1, 91.5, "90%", fontsize=8)
ax = axs[2]
s = df.sort_values("dt_max")
ax.bar(range(len(s)), s.dt_max, color=[FLAG if v > 60 else BLUE for v in s.dt_max], width=0.8)
ax.set_yscale("log"); ax.set_title("Largest gap per trip (s, log)", loc="left"); ax.set_xticks([])
ax.set_xlabel("trips, sorted"); ax.axhline(60, color=INK, lw=0.8, ls="--"); ax.text(1, 70, "60 s", fontsize=8)
for i, (_, r) in enumerate(s.tail(2).iterrows()):
    ax.annotate(f"trip {r.tid}: {r.dt_max / 3600:.1f} h", (len(s) - 2 + i, r.dt_max), xytext=(-95, -8 - 16 * (1 - i)),
                textcoords="offset points", fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_ylim(top=s.dt_max.max() * 5)
fig.tight_layout()
save("fig03_sampling")

# ---------- 4. speed ----------
spd = pools["spd"] * 3.6
spd = spd[np.isfinite(spd)]
fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(np.clip(spd, 0, 300), bins=np.arange(0, 302, 4), color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_yscale("log"); ax.set_xlabel("speed between consecutive fixes (km/h; values >300 shown at 300)")
ax.set_ylabel("steps (log)"); ax.set_title("Implied speed distribution, all 90 trips", loc="left")
ax.axvline(200, color=FLAG, lw=1.2, ls="--")
ax.text(203, ax.get_ylim()[1] * 0.3, f"{(spd > 200).sum():,} steps > 200 km/h\n"
        f"({(spd > 200).mean() * 100:.2f}% of all steps)", color=INK, fontsize=8)
save("fig04_speed")


# ---------- helpers for maps ----------
def trip_xy(part, t):
    _, tr, nd, ar, rt = load_trip(f"{BASE}/{part}/{t}")
    lon0, lat0 = np.median(tr[:, 0]), np.median(tr[:, 1])
    gx, gy = project(tr[:, 0], tr[:, 1], lon0, lat0)
    nx, ny = project(nd[:, 0], nd[:, 1], lon0, lat0)
    return tr, ar, rt, gx, gy, nx, ny


def draw_net(ax, ar, nx, ny, box, lw=0.5):
    x0, x1, y0, y1 = box
    fr, to = ar[:, 0], ar[:, 1]
    m = ((np.maximum(nx[fr], nx[to]) > x0) & (np.minimum(nx[fr], nx[to]) < x1) &
         (np.maximum(ny[fr], ny[to]) > y0) & (np.minimum(ny[fr], ny[to]) < y1))
    from matplotlib.collections import LineCollection
    segs = np.stack([np.c_[nx[fr[m]], ny[fr[m]]], np.c_[nx[to[m]], ny[to[m]]]], 1)
    ax.add_collection(LineCollection(segs, colors=NET, linewidths=lw, zorder=1))


def draw_route(ax, ar, rt, nx, ny, lw=2.2, z=2):
    from matplotlib.collections import LineCollection
    fr, to = ar[rt, 0], ar[rt, 1]
    segs = np.stack([np.c_[nx[fr], ny[fr]], np.c_[nx[to], ny[to]]], 1)
    ax.add_collection(LineCollection(segs, colors=GREEN, linewidths=lw, zorder=z, alpha=0.85))


def mapax(ax, box, title):
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.grid(False); ax.set_title(title, loc="left")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")


def map_legend(ax, loc="upper right", extra=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=NET, lw=1.5, label="road network arcs"),
         Line2D([], [], color=GREEN, lw=2.2, label="reference route (.route)"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="GPS fixes (.track)")]
    ax.legend(handles=h + (extra or []), loc=loc, fontsize=7.5, frameon=True, facecolor="white", edgecolor=GRID)


# ---------- 5. example trip ----------
T = "00000045"
tr, ar, rt, gx, gy, nx, ny = trip_xy("train", T)
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
pad = 300
box = (gx.min() - pad, gx.max() + pad, gy.min() - pad, gy.max() + pad)
ax = axs[0]
draw_net(ax, ar, nx, ny, box); draw_route(ax, ar, rt, nx, ny)
ax.scatter(gx, gy, s=2, color=BLUE, zorder=3)
ax.scatter(gx[[0, -1]], gy[[0, -1]], s=40, color=[AQUA, RED], zorder=4, edgecolor="white")
ax.text(gx[0] + 40, gy[0], "start", fontsize=8); ax.text(gx[-1] + 40, gy[-1], "end", fontsize=8)
mapax(ax, box, f"Train trip {T}: full extent")
map_legend(ax, "upper left")
# zoom on a segment with the biggest local network density
i = int(np.argmax(P(T, "n_cand50")))
cx, cy = gx[i], gy[i]
zb = (cx - 150, cx + 150, cy - 150, cy + 150)
ax.add_patch(Rectangle((zb[0], zb[2]), 300, 300, fill=False, ec=INK, lw=1, zorder=5))
ax = axs[1]
draw_net(ax, ar, nx, ny, zb, lw=1.2); draw_route(ax, ar, rt, nx, ny, lw=3)
m = (gx > zb[0]) & (gx < zb[1]) & (gy > zb[2]) & (gy < zb[3])
ax.plot(gx[m], gy[m], color=BLUE, lw=0.6, alpha=0.5, zorder=3)
ax.scatter(gx[m], gy[m], s=10, color=BLUE, zorder=4, edgecolor="white", linewidth=0.4)
fr, to = ar[:, 0], ar[:, 1]
mm = ((nx > zb[0]) & (nx < zb[1]) & (ny > zb[2]) & (ny < zb[3]))
ax.scatter(nx[mm], ny[mm], s=6, color=MUTED, zorder=2)
mapax(ax, zb, "Zoom (300 m box): GPS scatter around the route; grey dots = nodes")
save("fig05_example_trip")

# ---------- 6. anomalies: parked vehicle off-network ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, T in zip(axs, ["00000033", "00000054"]):
    tr, ar, rt, gx, gy, nx, ny = trip_xy("train", T)
    dw = P(T, "dwell")
    box = (-400, 400, -400, 400)
    draw_net(ax, ar, nx, ny, box, lw=0.8); draw_route(ax, ar, rt, nx, ny, lw=2.5)
    ax.scatter(gx[~dw], gy[~dw], s=3, color=BLUE, zorder=3)
    ax.scatter(gx[dw], gy[dw], s=3, color=FLAG, zorder=3, alpha=0.6)
    r = df[df.trip == T].iloc[0]
    mapax(ax, box, f"Train trip {T[-3:]}: {dw.mean() * 100:.0f}% of {len(gx):,} fixes are stationary")
    from matplotlib.lines import Line2D
    map_legend(ax, "lower left", [Line2D([], [], marker="o", ls="", color=FLAG, ms=4,
                                         label="stationary fixes (dwell >= 2 min)")])
save("fig06_anomaly_parked")

# trip 002: GPS continues past route end
T = "00000002"
tr, ar, rt, gx, gy, nx, ny = trip_xy("train", T)
d = P(T, "d_route")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
pad = 300
box = (gx.min() - pad, gx.max() + pad, gy.min() - pad, gy.max() + pad)
draw_net(ax, ar, nx, ny, box); draw_route(ax, ar, rt, nx, ny, lw=1.6, z=4)
ax.scatter(gx, gy, s=6, color=np.where(d > 50, FLAG, BLUE), zorder=3, alpha=0.5, linewidths=0)
mapax(ax, box, "Train trip 002: fixes > 50 m from the route in red")
map_legend(ax, "lower left")
ax = axs[1]
ax.plot(tr[:, 2] / 60, d, color=BLUE, lw=0.8)
ax.set_xlabel("time (min)"); ax.set_ylabel("distance to reference route (m)")
ax.set_title("Distance to route over time", loc="left")
save("fig07_anomaly_tail")

# ---------- 8. GPS-to-route distance ----------
keep = np.concatenate([P(t, "d_route")[~P(t, "dwell")] for t in df.trip])
allv = np.concatenate([P(t, "d_route") for t in df.trip])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axs[0]
ax.hist(np.clip(keep, 0, 60), bins=np.arange(0, 61, 1), color=BLUE, edgecolor="white", linewidth=0.3)
for q, lab, yy in [(50, "median", 0.92), (95, "p95", 0.75)]:
    v = np.percentile(keep, q)
    ax.axvline(v, color=INK, lw=0.8, ls="--")
    ax.text(v + 0.8, ax.get_ylim()[1] * yy, f"{lab} {v:.1f} m", fontsize=8)
ax.text(30, ax.get_ylim()[1] * 0.55, f"p99 = {np.percentile(keep, 99):.0f} m (off-axis)\n"
        f"bar at 60 = all fixes >= 60 m", fontsize=8, color=INK2)
ax.set_xlabel("distance from GPS fix to nearest reference-route arc (m, clipped at 60)")
ax.set_ylabel("fixes"); ax.set_title("GPS fix to reference route (stationary fixes removed)", loc="left")
ax = axs[1]
s = df.sort_values("d_route_p95_m").reset_index(drop=True)
x = np.arange(len(s))
ax.vlines(x, s.d_route_med_m, s.d_route_p95_m, color=NET, lw=2)
ax.scatter(x, s.d_route_med_m, s=14, color=[PCOL[p] for p in s.partition], zorder=3, label="median")
ax.scatter(x, s.d_route_p95_m, s=14, color="white", edgecolor=[PCOL[p] for p in s.partition], zorder=3)
ax.set_yscale("log"); ax.set_xticks([]); ax.set_xlabel("trips, sorted by p95 (filled = median, hollow = p95)")
ax.set_ylabel("metres (log)"); ax.set_title("Per-trip distance to route (all fixes)", loc="left")
for k, (i, r) in enumerate(s.tail(5).iterrows()):
    ax.annotate(r.tid, (i, r.d_route_p95_m), xytext=(-40, 12 + 13 * (k - 2)), textcoords="offset points", fontsize=7.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_ylim(top=s.d_route_p95_m.max() * 4)
part_legend(ax, "upper left")
fig.tight_layout()
save("fig08_dist_route")

# ---------- 9. road network ----------
fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
ax = axs[0, 0]
ax.scatter(df.n_nodes, df.n_arcs, s=16, color=[PCOL[p] for p in df.partition], edgecolor="white", linewidth=0.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("nodes"); ax.set_ylabel("arcs")
xx = np.array([200, 2e5]); ax.plot(xx, 2 * xx, color=MUTED, ls="--", lw=0.8); ax.text(400, 2500, "dashed: arcs = 2 x nodes", fontsize=8, color=INK2)
ax.set_title("Network size per trip", loc="left"); part_legend(ax, "lower right")
ax = axs[0, 1]
al = pools["alen"]
ax.hist(np.clip(al, 0.1, 3000), bins=np.logspace(-1, np.log10(3000), 50), color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xscale("log"); ax.set_xlabel("arc length (m, log)"); ax.set_ylabel("arcs")
ax.set_title(f"Arc length, all trips (median {np.median(al):.0f} m)", loc="left")
ax = axs[1, 0]
s = df.sort_values("largest_scc_frac")
ax.bar(range(len(s)), s.largest_scc_frac * 100, color=[FLAG if v < 0.9 else BLUE for v in s.largest_scc_frac], width=0.8)
ax.set_ylim(50, 101); ax.set_xticks([]); ax.set_xlabel("trips, sorted")
ax.set_ylabel("% of nodes"); ax.set_title("Share of nodes in the largest strongly connected component", loc="left")
ax = axs[1, 1]
s = df.sort_values("twin_arc_frac")
ax.bar(range(len(s)), s.twin_arc_frac * 100, color=BLUE, width=0.8)
ax.set_ylim(0, 101); ax.set_xticks([]); ax.set_xlabel("trips, sorted")
ax.set_ylabel("% of arcs"); ax.set_title("Arcs whose reverse arc also exists (two-way roads)", loc="left")
fig.tight_layout()
save("fig09_network")

# ---------- 10. label structure ----------
ppr = np.concatenate([P(t, "pts_per_route_pos") for t in df.trip if t not in ANOM])
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
ax = axs[0]
ax.hist(np.clip(ppr, 0, 40), bins=np.arange(0, 42, 1), color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("GPS fixes assigned to a route arc (clipped 40)"); ax.set_ylabel("route arcs")
ax.set_title(f"Fixes per route arc ({(ppr == 0).mean() * 100:.1f}% get none)", loc="left")
ax = axs[1]
s = df.sort_values("route_arcs_without_pts_frac")
ax.bar(range(len(s)), s.route_arcs_without_pts_frac * 100, color=[PCOL[p] for p in s.partition], width=0.8)
ax.set_xticks([]); ax.set_xlabel("trips, sorted"); ax.set_ylabel("% of route arcs")
ax.set_title("Route arcs with no GPS fix", loc="left")
ax = axs[2]
s = df.sort_values("naive_vs_mono_agree")
ax.bar(range(len(s)), s.naive_vs_mono_agree * 100, color=[PCOL[p] for p in s.partition], width=0.8)
ax.set_ylim(50, 101); ax.set_xticks([]); ax.set_xlabel("trips, sorted"); ax.set_ylabel("% of fixes")
ax.set_title("Simple vs order-aware labels agree", loc="left")
for i, (_, r) in enumerate(s.head(3).iterrows()):
    ax.text(i, r.naive_vs_mono_agree * 100 + 1, r.tid, rotation=90, fontsize=7, ha="center", va="bottom")
fig.tight_layout()
save("fig10_labels")

# ---------- 11. candidate generation ----------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
ax = axs[0]
R = np.arange(5, 201, 5)
for lab, ex, col in [("all fixes", set(), ORANGE), ("excluding trips 033 & 054", ANOM, BLUE)]:
    dtv = np.concatenate([P(t, "d_true") for t in df.trip if t not in ex])
    miss = [np.mean(~(dtv <= r)) * 100 for r in R]
    ax.plot(R, miss, color=col, lw=2, label=lab)
ax.set_yscale("log"); ax.set_xlabel("search radius r (m)"); ax.set_ylabel("% of fixes where it is missing (log)")
ax.set_title("Is the true arc inside the search circle?", loc="left"); ax.legend(fontsize=7.5)
ax = axs[1]
K = np.arange(1, 21)
rk = np.concatenate([P(t, "rank_geo") for t in df.trip if t not in ANOM])
rd = np.concatenate([P(t, "rank_dir") for t in df.trip if t not in ANOM])
ax.plot(K, [np.mean((rk >= 1) & (rk <= k)) * 100 for k in K], color=BLUE, lw=2, marker="o", ms=3,
        label="ignoring direction (twin arcs merged)")
mv = np.concatenate([P(t, "moving") for t in df.trip if t not in ANOM])
rd = rd[mv]
ax.plot(K, [np.mean((rd >= 1) & (rd <= k)) * 100 for k in K], color=ORANGE, lw=2, marker="o", ms=3,
        label="with direction, via a heading rule")
ax.set_xlabel("k nearest candidates"); ax.set_ylabel("% fixes with true arc in top-k")
ax.set_title("Is the true arc among the k nearest?", loc="left"); ax.legend(fontsize=7.5, loc="lower right")
ax.set_xticks([1, 2, 4, 6, 8, 10, 15, 20])
ax = axs[2]
nc = np.concatenate([P(t, "n_cand50") for t in df.trip if t not in ANOM])
ax.hist(np.clip(nc, 0, 60), bins=np.arange(0, 62, 2), color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("directed arcs within 50 m of a fix (clipped 60)"); ax.set_ylabel("fixes")
ax.set_title(f"Arcs within 50 m of a fix (median {np.median(nc):.0f})", loc="left")
fig.tight_layout()
save("fig11_candidates")

# ---------- 12. heading ----------
ht = np.concatenate([P(t, "head_err_true") for t in df.trip if t not in ANOM])
hw = np.concatenate([P(t, "head_err_twin") for t in df.trip if t not in ANOM])
ht, hw = ht[np.isfinite(ht)], hw[np.isfinite(hw)]
fig, ax = plt.subplots(figsize=(9, 3.2))
b = np.arange(0, 181, 5)
ax.hist(ht, bins=b, color=BLUE, alpha=0.85, label=f"vs true arc (median {np.median(ht):.0f} deg)", edgecolor="white", linewidth=0.3)
ax.hist(hw, bins=b, color=ORANGE, alpha=0.75, label=f"vs its reverse twin (median {np.median(hw):.0f} deg)", edgecolor="white", linewidth=0.3)
ax.set_xlabel("absolute angle between GPS heading and arc bearing (deg)"); ax.set_ylabel("moving fixes")
ax.set_title("Heading separates the two directions of a two-way road", loc="left"); ax.legend()
save("fig12_heading")

# ---------- 13. naive baseline ----------
s = df.sort_values("top1_nearest_geo")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(range(len(s)), s.top1_nearest_geo * 100, color=[PCOL[p] for p in s.partition], width=0.8)
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.tid, rotation=90, fontsize=6)
ax.set_ylabel("% fixes"); ax.set_ylim(0, 100)
med = df[~df.trip.isin(ANOM)].top1_nearest_geo.median() * 100
ax.axhline(med, color=INK, lw=0.8, ls="--"); ax.text(1, med + 2, f"median {med:.1f}%", fontsize=8)
ax.set_title("No-ML baseline: nearest road geometry = true road (per trip)", loc="left"); part_legend(ax, "lower right")
save("fig13_baseline")

# ---------- 14. pipeline diagram ----------
fig, ax = plt.subplots(figsize=(11, 2.3))
ax.set_xlim(0, 11); ax.set_ylim(0.55, 3.05); ax.axis("off")
boxes = [(0.1, "Raw trip\n.track .nodes\n.arcs .route", NET), (1.95, "Clean &\nproject\n(metres)", "#cde2fb"),
         (3.8, "Candidate\narcs per fix\n(KD-tree, r)", "#cde2fb"), (5.65, "Features +\nclassifier\nP(arc | fix)", "#b7d3f6"),
         (7.5, "HMM +\nViterbi over\nroad graph", "#9ec5f4"), (9.35, "Route +\nmetrics\n(API -> UI)", "#86b6ef")]
for x, txt, c in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.5), 1.5, 1.5, boxstyle="round,pad=0.03,rounding_size=0.12", fc=c, ec=MUTED))
    ax.text(x + 0.75, 2.25, txt, ha="center", va="center", fontsize=9)
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i + 1][0] - 0.03, 2.25), xytext=(boxes[i][0] + 1.53, 2.25),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
labels = ["labels: route ->\nper-fix arc (DP)", "drop parked tails,\nflag gaps", "miss rate vs r,\nrecall@k",
          ">= 5 models,\nGroupKFold by trip", "emission = P,\ntransition = path dist", "top-1, F1, continuity"]
for (x, _, _), l in zip(boxes, labels):
    ax.text(x + 0.75, 1.25, l, ha="center", va="top", fontsize=7.5, color=INK2)
save("fig14_pipeline")
print("figures written to", FIG)
