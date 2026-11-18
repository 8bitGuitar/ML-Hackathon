"""PathSnap data cleaning (steps C1-C10 of the briefing).

For every trip in train/ and student_test/ this writes
  data_clean/<partition>/<trip>/fixes.csv.gz   one row per GPS fix, with cleaned/derived columns and flags
  data_clean/<partition>/<trip>/arcs.csv.gz    one row per arc, with length/bearing and a candidate-eligible flag
and a summary data_clean/cleaning_summary.csv (one row per trip, counts per rule).

Rules that need the reference route (labels, C4, C6, C7) are computed only to build training data;
the prediction-time rules (C1-C3, C5, C8, C9) depend on GPS + map only, so the API can reuse them.

Usage: python clean.py "<Student folder>" <out dir>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_analysis"))
from analyze import dwell_mask, load_trip, monotone_labels, project, pt_seg_dist  # noqa: E402

# thresholds (chosen from train-data exploration, frozen here)
GAP_S = 60.0          # C2: start a new segment after a pause longer than this
JUMP_MS = 250 / 3.6   # C8: speed in AND out above this marks a position jump
FAR_M = 50.0          # C4/C6/C7: fixes further than this from their label are unreliable for training
DWELL_WIN_S, DWELL_RAD_M = 120, 30  # dwell definition
DWELL_KEEP_EVERY = 10  # C4: keep 1 in N fixes inside on-road dwells for training
MIN_MOVE_M = 4.0       # heading only defined when neighbours are this far apart


def clean_trip(part, d):
    t, track, nodes, arcs, route = load_trip(d)
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    n = len(track)
    # ---- C1: local metric projection (origin stored so the API can do the same) ----
    lon0, lat0 = float(np.median(lon)), float(np.median(lat))
    x, y = project(lon, lat, lon0, lat0)
    nx, ny = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    # ---- C3: real time differences and speeds ----
    dt = np.r_[np.nan, np.diff(ts)]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    speed_in = step / dt
    speed_out = np.r_[speed_in[1:], np.nan]
    # ---- C2: segments split at long pauses ----
    new_seg = np.r_[True, np.diff(ts) > GAP_S]
    segment = np.cumsum(new_seg) - 1
    # ---- heading from previous to next fix (within the same segment) ----
    ip = np.r_[0, np.arange(n - 1)]
    inx = np.r_[np.arange(1, n), n - 1]
    ip = np.where(segment[ip] == segment, ip, np.arange(n))
    inx = np.where(segment[inx] == segment, inx, np.arange(n))
    hx, hy = x[inx] - x[ip], y[inx] - y[ip]
    heading = np.degrees(np.arctan2(hx, hy)) % 360
    heading_ok = np.hypot(hx, hy) > MIN_MOVE_M
    # ---- dwell (stationary) detection ----
    dwell = dwell_mask(x, y, ts, DWELL_WIN_S, DWELL_RAD_M)
    # ---- C8: position jumps (implausible speed both into and out of the fix) ----
    jump = (np.nan_to_num(speed_in) > JUMP_MS) & (np.nan_to_num(speed_out) > JUMP_MS)
    # ---- C9: arcs eligible as candidates ----
    fr, to = arcs[:, 0], arcs[:, 1]
    alen = np.hypot(nx[to] - nx[fr], ny[to] - ny[fr])
    abear = np.degrees(np.arctan2(nx[to] - nx[fr], ny[to] - ny[fr])) % 360
    pairs = fr * len(nodes) + to
    has_twin = np.isin(to * len(nodes) + fr, pairs)
    dup = pd.Series(pairs).duplicated().values
    eligible = (fr != to) & (alen > 0.01)
    # ---- labels: order-aware assignment of fixes to route arcs ----
    rf, rt = fr[route], to[route]
    D, T = pt_seg_dist(x[:, None], y[:, None], nx[rf][None], ny[rf][None], nx[rt][None], ny[rt][None])
    D = D.astype(np.float32)
    pos = monotone_labels(D)
    d_label = D[np.arange(n), pos]
    label_arc = route[pos]
    d_route = D.min(1)
    # ---- training filters ----
    reason = np.full(n, "", dtype=object)

    def mark(mask, why):
        m = mask & (reason == "")
        reason[m] = why
        return int(m.sum())

    counts = {}
    counts["C8_jump"] = mark(jump, "C8_jump")
    counts["C4_offroad_dwell"] = mark(dwell & (d_route > FAR_M), "C4_offroad_dwell")
    # C7: overhang before route start / after route end (contiguous run of far fixes at the ends)
    far = d_label > FAR_M
    head = np.zeros(n, bool); tail = np.zeros(n, bool)
    i = 0
    while i < n and far[i] and pos[i] == 0:
        head[i] = True; i += 1
    i = n - 1
    while i >= 0 and far[i] and pos[i] == len(route) - 1:
        tail[i] = True; i -= 1
    counts["C7_beyond_route"] = mark(head | tail, "C7_beyond_route")
    counts["C6_far_from_label"] = mark(far, "C6_far_from_label")
    # C4: thin on-road dwells, keep every N-th fix of each dwell run
    run_id = np.cumsum(np.r_[True, dwell[1:] != dwell[:-1]])
    k_in_run = pd.Series(np.arange(n)).groupby(run_id).cumcount().values
    counts["C4_dwell_thinned"] = mark(dwell & (k_in_run % DWELL_KEEP_EVERY != 0), "C4_dwell_thinned")
    keep_train = reason == ""

    fixes = pd.DataFrame({
        "fix": np.arange(n), "lon": lon, "lat": lat, "t": ts, "x": x.round(3), "y": y.round(3),
        "dt": dt, "segment": segment, "speed_in": speed_in.round(3), "speed_out": speed_out.round(3),
        "heading": heading.round(2), "heading_ok": heading_ok, "dwell": dwell, "jump": jump,
        "label_arc": label_arc, "label_route_pos": pos, "d_label": d_label.round(3), "d_route": d_route.round(3),
        "keep_train": keep_train, "drop_reason": reason,
    })
    arcs_df = pd.DataFrame({"arc": np.arange(len(arcs)), "from": fr, "to": to, "length_m": alen.round(3),
                            "bearing": abear.round(2), "has_twin": has_twin, "duplicate": dup,
                            "candidate_eligible": eligible})
    od = os.path.join(OUT, part, t)
    os.makedirs(od, exist_ok=True)
    fixes.to_csv(f"{od}/fixes.csv.gz", index=False)
    arcs_df.to_csv(f"{od}/arcs.csv.gz", index=False)
    pd.Series({"lon0": lon0, "lat0": lat0}).to_json(f"{od}/projection.json")
    summary = dict(partition=part, trip=t, fixes=n, segments=int(segment.max() + 1),
                   dwell_fixes=int(dwell.sum()), jump_fixes=int(jump.sum()), **counts,
                   kept_for_training=int(keep_train.sum()),
                   arcs=len(arcs), arcs_excluded_C9=int((~eligible).sum()), duplicate_arcs=int(dup.sum()))
    return summary


if __name__ == "__main__":
    BASE, OUT = sys.argv[1], sys.argv[2]
    rows = []
    for part in ["train", "student_test"]:
        for d in sorted(glob.glob(f"{BASE}/{part}/*")):
            s = clean_trip(part, d)
            rows.append(s)
            print(part, s["trip"], f"kept {s['kept_for_training']}/{s['fixes']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/cleaning_summary.csv", index=False)
    print("done")
