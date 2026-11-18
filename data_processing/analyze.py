"""PathSnap dataset exploration.

Reads every trip in train/ and student_test/, computes GPS, road-network,
reference-route and GPS<->network statistics, and writes:
  out/trip_stats.csv      one row per trip
  out/point_sample.parquet-free npz of per-point diagnostics (for plotting)
Usage: python analyze.py "<path to Student folder>" <out dir>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

R_EARTH = 6371008.8
CAND_RADIUS = 200.0  # metres, search radius for candidate analysis
SAMPLE_STEP = 15.0   # metres, arc densification step for KD-tree


def load_trip(d):
    t = os.path.basename(d)
    rd = lambda ext, dt: np.loadtxt(f"{d}/{t}.{ext}", dtype=dt, ndmin=2)
    track = rd("track", float)
    nodes = rd("nodes", float)
    arcs = rd("arcs", np.int64)
    route = np.loadtxt(f"{d}/{t}.route", dtype=np.int64, ndmin=1)
    return t, track, nodes, arcs, route


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres around (lon0, lat0)."""
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def pt_seg_dist(px, py, ax, ay, bx, by):
    """Broadcasted point-to-segment distance; returns (dist, t in [0,1])."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def monotone_labels(D):
    """Assign each GPS point a route index, non-decreasing in time, minimising
    total point-to-arc distance (dynamic programming). D: (N, M) distances."""
    N, M = D.shape
    back = np.empty((N, M), dtype=np.int32)
    cost = D[0].astype(np.float64)
    back[0] = np.arange(M)
    idx = np.arange(M)
    for i in range(1, N):
        # prefix minimum of previous cost and where it was attained
        pm = np.minimum.accumulate(cost)
        is_new = np.r_[True, cost[1:] < pm[:-1]]
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        back[i] = arg
        cost = pm + D[i]
    lab = np.empty(N, dtype=np.int64)
    lab[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        lab[i - 1] = back[i, lab[i]]
    return lab


def dwell_mask(gx, gy, ts, win=120, rad=30):
    """Mark points inside any >=win-second window whose points all stay within
    rad metres of the window centroid (vehicle parked / stopped)."""
    n = len(gx)
    mark = np.zeros(n, bool)
    j = 0
    for i in range(n):
        while j < n and ts[j] - ts[i] <= win:
            j += 1
        if ts[j - 1] - ts[i] >= win * 0.9:
            sx, sy = gx[i:j], gy[i:j]
            if np.hypot(sx - sx.mean(), sy - sy.mean()).max() < rad:
                mark[i:j] = True
    return mark


def bearing(ax, ay, bx, by):
    return np.degrees(np.arctan2(bx - ax, by - ay)) % 360


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def analyse_trip(part, d):
    t, track, nodes, arcs, route = load_trip(d)
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    lon0, lat0 = np.median(lon), np.median(lat)
    gx, gy = project(lon, lat, lon0, lat0)
    nx, ny = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    n_nodes, n_arcs, n_pts = len(nodes), len(arcs), len(track)
    s = dict(partition=part, trip=t, n_points=n_pts, n_nodes=n_nodes, n_arcs=n_arcs,
             n_route_arcs=len(route), lon_c=lon0, lat_c=lat0)

    # ---------------- GPS track ----------------
    dt = np.diff(ts)
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dwell = dwell_mask(gx, gy, ts)
    spd = np.where(dt > 0, step / np.where(dt > 0, dt, 1), np.nan)
    s.update(
        t_start=ts[0], duration_s=ts[-1] - ts[0],
        nan_values=int(np.isnan(track).sum()),
        ts_nonint=int((np.abs(ts - np.round(ts)) > 1e-6).sum()),
        ts_nonmonotonic=int((dt <= 0).sum()),
        dt_frac_1s=float(np.mean(np.abs(dt - 1) < 1e-6)),
        dt_median=float(np.median(dt)), dt_max=float(dt.max()),
        gaps_gt10s=int((dt > 10).sum()), gaps_gt60s=int((dt > 60).sum()),
        dup_consecutive=int(((step < 1e-6)).sum()),
        stationary_frac=float(np.mean(step / np.maximum(dt, 1) < 0.5)),
        dwell_frac=float(dwell.mean()),
        path_len_km=float(step.sum() / 1000),
        speed_med_kmh=float(np.nanmedian(spd) * 3.6),
        speed_p99_kmh=float(np.nanpercentile(spd, 99) * 3.6),
        speed_max_kmh=float(np.nanmax(spd) * 3.6),
        speed_gt200_steps=int(np.nansum(spd * 3.6 > 200)),
        extent_ew_km=float((gx.max() - gx.min()) / 1000),
        extent_ns_km=float((gy.max() - gy.min()) / 1000),
        bad_coords=int(((np.abs(lat) > 90) | (np.abs(lon) > 180) | ((lat == 0) & (lon == 0))).sum()),
    )

    # ---------------- Road network ----------------
    fr, to = arcs[:, 0], arcs[:, 1]
    valid_ids = (fr >= 0) & (fr < n_nodes) & (to >= 0) & (to < n_nodes)
    alen = np.hypot(nx[to] - nx[fr], ny[to] - ny[fr])
    pairs = fr * n_nodes + to
    rev = to * n_nodes + fr
    has_twin = np.isin(rev, pairs)
    used = np.zeros(n_nodes, bool)
    used[fr] = used[to] = True
    G = coo_matrix((np.ones(n_arcs), (fr, to)), shape=(n_nodes, n_nodes)).tocsr()
    nw, wlab = connected_components(G, directed=True, connection="weak")
    ns, slab = connected_components(G, directed=True, connection="strong")
    _, wsz = np.unique(wlab, return_counts=True)
    _, ssz = np.unique(slab, return_counts=True)
    outdeg = np.bincount(fr, minlength=n_nodes)
    indeg = np.bincount(to, minlength=n_nodes)
    dup_nodes = int(pd.DataFrame(nodes).duplicated().sum())
    s.update(
        arc_ids_valid=bool(valid_ids.all()),
        self_loops=int((fr == to).sum()),
        dup_arcs=int(pd.Index(pairs).duplicated().sum()),
        dup_node_coords=dup_nodes,
        isolated_nodes=int((~used).sum()),
        twin_arc_frac=float(has_twin.mean()),
        arc_len_med_m=float(np.median(alen)), arc_len_p95_m=float(np.percentile(alen, 95)),
        arc_len_max_m=float(alen.max()), zero_len_arcs=int((alen < 0.01).sum()),
        weak_components=int(nw), strong_components=int(ns),
        largest_wcc_frac=float(wsz.max() / n_nodes), largest_scc_frac=float(ssz.max() / n_nodes),
        dead_end_nodes=int(((outdeg == 0) & (indeg > 0)).sum()),
        source_only_nodes=int(((indeg == 0) & (outdeg > 0)).sum()),
        net_extent_ew_km=float((nx.max() - nx.min()) / 1000),
        net_extent_ns_km=float((ny.max() - ny.min()) / 1000),
        gps_outside_net_bbox=int(((gx < nx.min()) | (gx > nx.max()) | (gy < ny.min()) | (gy > ny.max())).sum()),
    )

    # ---------------- Reference route ----------------
    r_ok = (route >= 0) & (route < n_arcs)
    s["route_ids_valid"] = bool(r_ok.all())
    rf, rt = fr[route], to[route]
    breaks = rt[:-1] != rf[1:]
    s.update(
        route_breaks=int(breaks.sum()),
        route_unique_arcs=int(len(np.unique(route))),
        route_repeated_arcs=int(len(route) - len(np.unique(route))),
        route_immediate_uturns=int(((rf[1:] == rt[:-1]) & (rt[1:] == rf[:-1])).sum()),
        route_len_km=float(alen[route].sum() / 1000),
        route_in_largest_scc=bool(np.all(slab[rf] == np.argmax(np.bincount(slab)))),
    )

    # ---------------- GPS <-> route distances & per-point labels ----------------
    ax, ay, bx, by = nx[rf], ny[rf], nx[rt], ny[rt]
    D, _ = pt_seg_dist(gx[:, None], gy[:, None], ax[None], ay[None], bx[None], by[None])
    D = D.astype(np.float32)
    d_route = D.min(1)
    naive_idx = D.argmin(1)
    mono_idx = monotone_labels(D)
    d_mono = D[np.arange(n_pts), mono_idx]
    lab_arc = route[mono_idx]
    pts_per_route_pos = np.bincount(mono_idx, minlength=len(route))
    s.update(
        d_route_med_m=float(np.median(d_route)), d_route_p95_m=float(np.percentile(d_route, 95)),
        d_route_max_m=float(d_route.max()),
        frac_pts_gt30m_route=float(np.mean(d_route > 30)),
        frac_pts_gt100m_route=float(np.mean(d_route > 100)),
        d_mono_med_m=float(np.median(d_mono)), d_mono_p95_m=float(np.percentile(d_mono, 95)),
        naive_vs_mono_agree=float(np.mean(route[naive_idx] == lab_arc)),
        route_arcs_without_pts_frac=float(np.mean(pts_per_route_pos == 0)),
        n_label_classes=int(len(np.unique(lab_arc))),
        start_gap_m=float(d_route[0]), end_gap_m=float(d_route[-1]),
        first_label_pos=int(mono_idx[0]), last_label_pos=int(mono_idx[-1]),
    )

    # ---------------- candidate analysis (whole network) ----------------
    # restrict to arcs near the track
    pad = CAND_RADIUS + 50
    near = ((np.maximum(nx[fr], nx[to]) > gx.min() - pad) & (np.minimum(nx[fr], nx[to]) < gx.max() + pad) &
            (np.maximum(ny[fr], ny[to]) > gy.min() - pad) & (np.minimum(ny[fr], ny[to]) < gy.max() + pad))
    near_ids = np.flatnonzero(near)
    # densify arcs into sample points
    nseg = np.maximum(1, np.ceil(alen[near_ids] / SAMPLE_STEP).astype(int))
    rep = np.repeat(near_ids, nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
    sx = nx[fr[rep]] + frac * (nx[to[rep]] - nx[fr[rep]])
    sy = ny[fr[rep]] + frac * (ny[to[rep]] - ny[fr[rep]])
    tree = cKDTree(np.c_[sx, sy])
    hits = tree.query_ball_point(np.c_[gx, gy], r=CAND_RADIUS + SAMPLE_STEP)
    # GPS heading (central difference, only where moving)
    hx = np.r_[gx[1:], gx[-1]] - np.r_[gx[0], gx[:-1]]
    hy = np.r_[gy[1:], gy[-1]] - np.r_[gy[0], gy[:-1]]
    moving = np.hypot(hx, hy) > 4
    gps_head = np.degrees(np.arctan2(hx, hy)) % 360
    arc_head = bearing(nx[fr], ny[fr], nx[to], ny[to])

    rank_geo = np.full(n_pts, -1)      # rank of true arc among candidates by distance (twins merged)
    rank_dir = np.full(n_pts, -1)      # rank by distance with wrong-direction arcs penalised
    n_cand50 = np.zeros(n_pts, int)
    n_cand100 = np.zeros(n_pts, int)
    d_true = np.full(n_pts, np.nan)
    head_err_true = np.full(n_pts, np.nan)
    head_err_twin = np.full(n_pts, np.nan)
    true_has_twin = has_twin[lab_arc]
    for i in range(n_pts):
        cand = np.unique(rep[hits[i]])
        if cand.size == 0:
            continue
        dd, _ = pt_seg_dist(gx[i], gy[i], nx[fr[cand]], ny[fr[cand]], nx[to[cand]], ny[to[cand]])
        keep = dd <= CAND_RADIUS
        cand, dd = cand[keep], dd[keep]
        n_cand50[i] = int((dd <= 50).sum())
        n_cand100[i] = int((dd <= 100).sum())
        tgt = lab_arc[i]
        if moving[i]:
            head_err_true[i] = angdiff(gps_head[i], arc_head[tgt])
            if true_has_twin[i]:
                head_err_twin[i] = angdiff(gps_head[i], (arc_head[tgt] + 180) % 360)
        pos = np.flatnonzero(cand == tgt)
        if pos.size == 0:
            continue
        dt_ = dd[pos[0]]
        d_true[i] = dt_
        # undirected geometry: merge twin pairs -> key = sorted node pair
        key = np.minimum(fr[cand], to[cand]) * n_nodes + np.maximum(fr[cand], to[cand])
        tkey = min(fr[tgt], to[tgt]) * n_nodes + max(fr[tgt], to[tgt])
        ukeys = pd.Series(dd).groupby(key).min()
        rank_geo[i] = int((ukeys.values < ukeys.loc[tkey] - 1e-9).sum()) + 1
        if moving[i]:
            pen = dd + 1000 * (angdiff(gps_head[i], arc_head[cand]) > 90)
            rank_dir[i] = int((pen < pen[pos[0]] - 1e-9).sum()) + 1

    found = ~np.isnan(d_true)
    s.update(
        cand_miss_200m=float(np.mean(~found)),
        cand_med_within50=float(np.median(n_cand50)), cand_med_within100=float(np.median(n_cand100)),
        top1_nearest_geo=float(np.mean(rank_geo == 1)),
        top1_nearest_dir=float(np.mean(rank_dir[moving] == 1)) if moving.any() else np.nan,
        true_has_twin_frac=float(true_has_twin.mean()),
    )
    per_point = dict(d_route=d_route, d_true=d_true, rank_geo=rank_geo, rank_dir=rank_dir,
                     n_cand50=n_cand50, n_cand100=n_cand100, head_err_true=head_err_true,
                     head_err_twin=head_err_twin, dt=np.r_[np.nan, dt], speed=np.r_[np.nan, spd],
                     moving=moving, dwell=dwell, mono_idx=mono_idx, pts_per_route_pos=pts_per_route_pos)
    return s, per_point, dict(dt=dt, alen=alen, spd=spd)


def main():
    base, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows, pp = [], {}
    pools = {"dt": [], "alen": [], "spd": []}
    for part in ["train", "student_test"]:
        for d in sorted(glob.glob(f"{base}/{part}/*")):
            s, p, pool = analyse_trip(part, d)
            rows.append(s)
            pp[s["trip"]] = p
            for k in pools:
                pools[k].append(pool[k])
            print(part, s["trip"], s["n_points"], f"miss={s['cand_miss_200m']:.3f}",
                  f"top1={s['top1_nearest_geo']:.3f}", f"breaks={s['route_breaks']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/trip_stats.csv", index=False)
    flat = {}
    for t, p in pp.items():
        for k, v in p.items():
            flat[f"{t}__{k}"] = v
    np.savez_compressed(f"{out}/per_point.npz", **flat)
    np.savez_compressed(f"{out}/pools.npz", **{k: np.concatenate(v) for k, v in pools.items()})
    all_ids = set(f"{i:08d}" for i in range(100))
    json.dump({"missing_trip_ids": sorted(all_ids - set(df.trip))}, open(f"{out}/meta.json", "w"))


if __name__ == "__main__":
    main()
