"""Route prediction for one unseen trip (GPS + map only, the reference route is never read).

    from predict import match_trip
    result = match_trip("Student folder/student_test/00000007")

Pipeline, in order (same code and constants as the notebooks; see pipeline/clean.py and feature_engineering.ipynb):
  1. cleaning that needs only GPS and map: projection (C1), time gaps / segments (C2), speeds (C3), heading, dwell
     detection, position jumps (C8), eligible arcs (C9)
  2. candidate search: eligible arcs within 50 m of each fix (at least 5 per fix)
  3. features F1-F14 (the 13 selected ones)
  4. tuned HistGradientBoosting classifier -> probability per candidate
  5. HMM / Viterbi (pathsnap_hmm.py) -> one arc per fix, then a connected arc sequence = the route
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config

for _p in (config.PIPELINE_DIR, config.ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze import dwell_mask, project  # noqa: E402
from pathsnap_hmm import TripNet, prepare_trip, reconstruct_route, viterbi_trip  # noqa: E402

# constants frozen in the cleaning / feature notebooks
GAP_S = 60.0
JUMP_MS = 250 / 3.6
DWELL_WIN_S, DWELL_RAD_M = 120, 30
MIN_MOVE_M = 4.0
SAMPLE_STEP = 10.0


# ------------------------------------------------------------------------------------------------ model loading
@lru_cache(maxsize=1)
def load_models():
    """Classifier + settings, loaded once per process."""
    clf_cfg = json.load(open(config.MODEL_DIR / "classifier_config.json"))
    hmm_cfg = json.load(open(config.MODEL_DIR / "hmm_config.json"))
    clf = joblib.load(config.MODEL_DIR / "classifier.joblib")
    return clf, clf_cfg, hmm_cfg


# ------------------------------------------------------------------------------------------------ raw input
def load_raw_trip(trip_dir):
    """Reads .track / .nodes / .arcs only. The .route file (ground truth) is deliberately not touched."""
    d = Path(trip_dir)
    t = d.name
    track = np.loadtxt(d / f"{t}.track", dtype=float, ndmin=2)     # lon, lat, time
    nodes = np.loadtxt(d / f"{t}.nodes", dtype=float, ndmin=2)     # lon, lat
    arcs = np.loadtxt(d / f"{t}.arcs", dtype=np.int64, ndmin=2)    # from, to
    return track, nodes, arcs


# ------------------------------------------------------------------------------------------------ step 1: cleaning
def prepare_fixes_and_arcs(track, nodes, arcs):
    """Prediction-time cleaning (C1-C3, C8, C9). Returns fixes, arcs tables, projected node coordinates."""
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    n = len(track)
    lon0, lat0 = float(np.median(lon)), float(np.median(lat))
    x, y = project(lon, lat, lon0, lat0)
    nx, ny = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    dt = np.r_[np.nan, np.diff(ts)]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    speed_in = step / dt
    speed_out = np.r_[speed_in[1:], np.nan]
    segment = np.cumsum(np.r_[True, np.diff(ts) > GAP_S]) - 1
    ip = np.r_[0, np.arange(n - 1)]
    inx = np.r_[np.arange(1, n), n - 1]
    ip = np.where(segment[ip] == segment, ip, np.arange(n))
    inx = np.where(segment[inx] == segment, inx, np.arange(n))
    hx, hy = x[inx] - x[ip], y[inx] - y[ip]
    heading = np.degrees(np.arctan2(hx, hy)) % 360
    heading_ok = np.hypot(hx, hy) > MIN_MOVE_M
    dwell = dwell_mask(x, y, ts, DWELL_WIN_S, DWELL_RAD_M)
    jump = (np.nan_to_num(speed_in) > JUMP_MS) & (np.nan_to_num(speed_out) > JUMP_MS)
    fixes = pd.DataFrame({"fix": np.arange(n), "lon": lon, "lat": lat, "t": ts, "x": x, "y": y, "dt": dt,
                          "segment": segment, "speed_in": speed_in, "speed_out": speed_out, "heading": heading,
                          "heading_ok": heading_ok, "dwell": dwell, "jump": jump})
    fr, to = arcs[:, 0], arcs[:, 1]
    alen = np.hypot(nx[to] - nx[fr], ny[to] - ny[fr])
    abear = np.degrees(np.arctan2(nx[to] - nx[fr], ny[to] - ny[fr])) % 360
    pairs = fr * len(nodes) + to
    has_twin = np.isin(to * len(nodes) + fr, pairs)
    eligible = (fr != to) & (alen > 0.01)
    arcs_df = pd.DataFrame({"arc": np.arange(len(arcs)), "from": fr, "to": to, "length_m": alen, "bearing": abear,
                            "has_twin": has_twin, "candidate_eligible": eligible})
    return fixes, arcs_df, nx, ny


# ------------------------------------------------------------------------------------------------ step 2: candidates
def _point_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def _angle_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def generate_candidates(fx, nx, ny, arcs_df, r_cand, k_min):
    """(fix, arc, distance, t_along) for every candidate pair: eligible arcs within r_cand m, topped up to k_min."""
    gx, gy = fx.x.values, fx.y.values
    fr, to = arcs_df["from"].values, arcs_df["to"].values
    pad = 350
    near = (arcs_df.candidate_eligible.values &
            (np.maximum(nx[fr], nx[to]) > gx.min() - pad) & (np.minimum(nx[fr], nx[to]) < gx.max() + pad) &
            (np.maximum(ny[fr], ny[to]) > gy.min() - pad) & (np.minimum(ny[fr], ny[to]) < gy.max() + pad))
    ids = np.flatnonzero(near)
    nseg = np.maximum(1, np.ceil(arcs_df.length_m.values[ids] / SAMPLE_STEP).astype(int))
    owner = np.repeat(ids, nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
    sx = nx[fr[owner]] + frac * (nx[to[owner]] - nx[fr[owner]])
    sy = ny[fr[owner]] + frac * (ny[to[owner]] - ny[fr[owner]])
    tree = cKDTree(np.c_[sx, sy])

    def pairs_within(fix_idx, radius):
        hits = tree.query_ball_point(np.c_[gx[fix_idx], gy[fix_idx]], r=radius + SAMPLE_STEP / 2)
        lens = np.fromiter((len(h) for h in hits), int, len(hits))
        fi = np.repeat(fix_idx, lens)
        ai = owner[np.concatenate(hits).astype(int)] if lens.sum() else np.array([], int)
        key = np.unique(fi.astype(np.int64) * len(arcs_df) + ai)
        fi, ai = key // len(arcs_df), key % len(arcs_df)
        d, t = _point_segment_distance(gx[fi], gy[fi], nx[fr[ai]], ny[fr[ai]], nx[to[ai]], ny[to[ai]])
        return fi, ai, d, t

    fi, ai, d, t = pairs_within(np.arange(len(fx)), r_cand)
    m = d <= r_cand
    fi, ai, d, t = fi[m], ai[m], d[m], t[m]
    counts = np.bincount(fi, minlength=len(fx))
    short = np.flatnonzero(counts < k_min)
    if len(short):
        f2, a2, d2, t2 = pairs_within(short, 300.0)
        extra = pd.DataFrame({"f": f2, "a": a2, "d": d2, "t": t2}).sort_values(["f", "d"])
        extra = extra.groupby("f").head(k_min)
        extra = extra[extra.d > r_cand]
        fi = np.r_[fi, extra.f.values]; ai = np.r_[ai, extra.a.values]
        d = np.r_[d, extra.d.values]; t = np.r_[t, extra.t.values]
    return fi, ai, d, t


# ------------------------------------------------------------------------------------------------ step 3: features
def build_features(fx, arcs_df, nx, ny, n_nodes, r_cand, k_min, timing=None):
    """One row per (fix, candidate arc) with features F1-F14 (no label columns)."""
    fr, to = arcs_df["from"].values, arcs_df["to"].values
    t0 = time.perf_counter()
    fi, ai, d, t = generate_candidates(fx, nx, ny, arcs_df, r_cand, k_min)
    if timing is not None:
        timing["candidates_s"] = time.perf_counter() - t0
    c = pd.DataFrame({"fix": fi, "arc": ai, "dist": d, "along_t": t})
    c["dist_rank"] = c.groupby("fix").dist.rank(method="first").astype(int)
    c["dist_gap"] = c.dist - c.groupby("fix").dist.transform("min")
    c["n_cand20"] = (c.dist <= 20).groupby(c.fix).transform("sum")
    c["n_cand50"] = (c.dist <= 50).groupby(c.fix).transform("sum")

    f = fx.iloc[c.fix.values]
    moving = f.heading_ok.values & ~f.dwell.values
    hd = _angle_diff(f.heading.values, arcs_df.bearing.values[c.arc.values])
    c["heading_diff"] = np.where(moving, hd, 90.0)                   # 90 = "no information"
    c["speed"] = np.nan_to_num(np.nanmean(np.c_[f.speed_in.values, f.speed_out.values], axis=1))

    deg = np.bincount(fr, minlength=n_nodes) + np.bincount(to, minlength=n_nodes)
    c["arc_len"] = arcs_df.length_m.values[c.arc.values]
    c["has_twin"] = arcs_df.has_twin.values[c.arc.values].astype(int)
    c["junction_deg"] = deg[fr[c.arc.values]] + deg[to[c.arc.values]]

    seg, n = fx.segment.values, len(fx)
    prev_i = np.where((c.fix.values > 0) & (seg[np.maximum(c.fix.values - 1, 0)] == seg[c.fix.values]),
                      c.fix.values - 1, c.fix.values)
    next_i = np.where((c.fix.values < n - 1) & (seg[np.minimum(c.fix.values + 1, n - 1)] == seg[c.fix.values]),
                      c.fix.values + 1, c.fix.values)
    a_fr, a_to = fr[c.arc.values], to[c.arc.values]
    dp, _ = _point_segment_distance(fx.x.values[prev_i], fx.y.values[prev_i], nx[a_fr], ny[a_fr], nx[a_to], ny[a_to])
    dn, _ = _point_segment_distance(fx.x.values[next_i], fx.y.values[next_i], nx[a_fr], ny[a_fr], nx[a_to], ny[a_to])
    c["nbr_dist"] = (dp + dn) / 2

    nearest = c[c.dist_rank == 1].set_index("fix").arc
    prev_best = nearest.reindex(prev_i).values
    has_prev = (prev_i != c.fix.values) & ~pd.isna(prev_best)
    pb = np.where(has_prev, prev_best, 0).astype(int)
    same = c.arc.values == pb
    follows = fr[c.arc.values] == to[pb]
    shares = (fr[c.arc.values] == fr[pb]) | (to[c.arc.values] == to[pb]) | (to[c.arc.values] == fr[pb])
    conn = np.where(same, 0, np.where(follows, 1, np.where(shares, 2, 3)))
    c["conn_prev"] = np.where(has_prev, conn, 1)
    return c


# ------------------------------------------------------------------------------------------------ steps 4-5 + output
def match_arrays(track, nodes, arcs):
    """Full pipeline on in-memory arrays. Returns dict(fix_arc, route, holes, breaks, n_candidates)."""
    clf, clf_cfg, hmm_cfg = load_models()
    timing = {}
    t0 = time.perf_counter()
    fx, arcs_df, nx, ny = prepare_fixes_and_arcs(track, nodes, arcs)
    timing["cleaning_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cand = build_features(fx, arcs_df, nx, ny, len(nodes), clf_cfg["candidate_radius_m"], clf_cfg["min_candidates"],
                          timing)                                    # sets timing["candidates_s"]
    timing["features_s"] = time.perf_counter() - t0 - timing["candidates_s"]
    t0 = time.perf_counter()
    p = clf.predict_proba(cand[clf_cfg["features"]].to_numpy(np.float32))[:, 1]
    timing["classifier_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()

    net = TripNet(arrays=(arcs_df["from"].values, arcs_df["to"].values, arcs_df.length_m.values.astype(float),
                          arcs_df.candidate_eligible.values))
    prep = prepare_trip(None, None, cand, p, net=net, limit=hmm_cfg["limit"], fx=fx)
    pred, breaks, _ = viterbi_trip(prep, beta=hmm_cfg["beta"], vmax=hmm_cfg["vmax"], slack=hmm_cfg["slack"],
                                   back_tol=hmm_cfg["back_tol"], p_floor=hmm_cfg["p_floor"])
    route, holes = reconstruct_route(net, pred, limit=hmm_cfg["limit"])
    timing["hmm_s"] = time.perf_counter() - t0
    return dict(fix_arc=pred, route=route, holes=int(holes), breaks=int(breaks), n_candidates=len(cand),
                timing=timing)


def route_polyline(route, nodes, arcs):
    """Arc sequence -> list of [lat, lon] points (consecutive arcs share a node, so it is one connected line)."""
    if len(route) == 0:
        return []
    seq = [arcs[route[0], 0]] + [arcs[a, 1] for a in route]
    return [[float(nodes[n, 1]), float(nodes[n, 0])] for n in seq]


def match_trip(trip_dir):
    """Raw trip folder -> matched route.

    Returns {"route": [arc ids], "polyline": [[lat, lon], ...], "track": [[lat, lon], ...],
             "fix_arc": [arc id per fix], "holes", "breaks", "n_fixes"}
    """
    track, nodes, arcs = load_raw_trip(trip_dir)
    r = match_arrays(track, nodes, arcs)
    return dict(route=r["route"].tolist(), polyline=route_polyline(r["route"], nodes, arcs),
                track=track[:, [1, 0]].tolist(), fix_arc=r["fix_arc"].tolist(), holes=r["holes"],
                breaks=r["breaks"], n_fixes=len(track), n_candidates=r["n_candidates"], timing=r["timing"])


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = match_trip(sys.argv[1])
    print(f"{out['n_fixes']} fixes -> {len(out['route'])} route arcs in {time.time() - t0:.1f}s "
          f"(breaks {out['breaks']}, holes {out['holes']})")
