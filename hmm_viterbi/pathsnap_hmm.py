"""HMM + Viterbi route reconstruction for PathSnap.

Hidden states   : the candidate arcs of each GPS fix (state j of fix k = "the car is on candidate arc j at the
                  point closest to fix k"), so every state maps back to exactly one road segment.
Emission        : log of the classifier probability of the candidate, normalised over the fix's candidates.
Transition      : Newson & Krumm (2009) style. d_route = driving distance along the directed network from the
                  previous state's position to this state's position; d_line = straight-line distance between the
                  two fixes. log T = -|d_route - d_line| / beta. A transition is impossible (-inf) when d_route
                  is longer than the car could drive in the time between fixes (vmax * dt + slack).
Decoding        : Viterbi (max-sum dynamic programming) with back-pointers.
Breaks          : the chain is cut and restarted (a) at segment boundaries (time gap > 60 s, cleaning step C2) and
                  (b) when no candidate of the new fix is reachable from any candidate of the previous one.
Skipped fixes   : position jumps (cleaning C8) and fixes without candidates are not decoded; they inherit the
                  arc of the nearest decoded fix before them (or after, at the start).
Thinning        : a fix closer than min_move metres to the last decoded fix is not decoded (Newson & Krumm drop
                  such fixes too). GPS noise here is several metres per fix, so between two fixes one second apart
                  the noise is as large as the real movement and d_line says little; while the car is parked the
                  fixes only jitter. Decoding one fix every min_move metres keeps the transition term meaningful.
Route           : reconstruct_route joins the decoded arcs into one connected arc sequence (shortest-path fill).
Snapping        : snap_to_route then gives EVERY fix (also the thinned and skipped ones) its arc on that route:
                  fixes are assigned to route positions in travel order (never going back along the route) so
                  that the total fix-to-arc distance is smallest. decode_trip runs the three steps in order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

for _d in ("model_training", "data_processing"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / _d))
from analyze import pt_seg_dist  # noqa: E402
from pathsnap_common import load_arcs, load_fixes  # noqa: E402

DEFAULTS = dict(beta=5.0, vmax=50.0, slack=30.0, back_tol=15.0, limit=2500.0, p_floor=1e-6, min_move=0.0)


class TripNet:
    """Directed road graph of one trip (eligible arcs only) with shortest-path helpers."""

    def __init__(self, part=None, tidx=None, arrays=None):
        # arrays=(from, to, length, eligible) lets the API build the graph from a raw trip (no data_clean/)
        fr, to, length, elig = arrays if arrays is not None else load_arcs(part, tidx)
        self.fr, self.to, self.len = fr, to, length
        self.n = int(max(fr.max(), to.max())) + 1
        e = np.flatnonzero(elig & (fr != to))
        e = e[np.argsort(length[e], kind="stable")]
        key = fr[e].astype(np.int64) * self.n + to[e]
        _, first = np.unique(key, return_index=True)       # shortest arc for each (from, to) node pair
        e = e[first]
        self.pair_arc = dict(zip((fr[e].astype(np.int64) * self.n + to[e]).tolist(), e.tolist()))
        self.G = csr_matrix((length[e], (fr[e], to[e])), shape=(self.n, self.n))

    def distance_table(self, src, dst, limit, chunk=64):
        """Shortest driving distance from every node in src to every node in dst (inf if > limit)."""
        src_u, dst_u = np.unique(src), np.unique(dst)
        D = np.full((len(src_u), len(dst_u)), np.inf, dtype=np.float32)
        for i in range(0, len(src_u), chunk):
            d = dijkstra(self.G, indices=src_u[i:i + chunk], limit=limit)
            D[i:i + chunk] = d[:, dst_u]
        return src_u, dst_u, D

    def path_arcs(self, a, b, limit):
        """Arcs on the shortest path from the end of arc a to the start of arc b (empty if directly connected,
        None if unreachable within limit)."""
        s, t = self.to[a], self.fr[b]
        if s == t:
            return []
        dist, pred = dijkstra(self.G, indices=s, limit=limit, return_predecessors=True)
        if not np.isfinite(dist[t]):
            return None
        nodes = [t]
        while nodes[-1] != s:
            nodes.append(pred[nodes[-1]])
        nodes = nodes[::-1]
        return [self.pair_arc[int(u) * self.n + int(v)] for u, v in zip(nodes[:-1], nodes[1:])]


def _lookup(table, s_nodes, d_nodes):
    src_u, dst_u, D = table
    return D[np.searchsorted(src_u, s_nodes)[:, None], np.searchsorted(dst_u, d_nodes)[None, :]]


def prepare_trip(part, tidx, rows, p, net=None, limit=DEFAULTS["limit"], fx=None):
    """Group candidate rows of one trip by fix and precompute the distance table (independent of beta)."""
    net = net or TripNet(part, tidx)
    fx = load_fixes(part)[tidx] if fx is None else fx
    order = np.argsort(rows.fix.values, kind="stable")
    fix = rows.fix.values[order]
    arc = rows.arc.values[order].astype(np.int64)
    t = rows.along_t.values[order].astype(float)
    pp = p[order].astype(float)
    starts = np.flatnonzero(np.r_[True, fix[1:] != fix[:-1]])
    ends = np.r_[starts[1:], len(fix)]
    cands = {int(fix[s]): (arc[s:e], t[s:e], pp[s:e]) for s, e in zip(starts, ends)}
    ua = np.unique(arc)
    table = net.distance_table(net.to[ua], np.r_[net.fr[ua], net.fr[ua]], limit)
    return dict(net=net, fx=fx, cands=cands, table=table)


def viterbi_trip(prep, beta=DEFAULTS["beta"], vmax=DEFAULTS["vmax"], slack=DEFAULTS["slack"],
                 back_tol=DEFAULTS["back_tol"], p_floor=DEFAULTS["p_floor"], min_move=DEFAULTS["min_move"]):
    """Decode one trip. Returns (predicted arc per fix, number of chain breaks, decoded-fix mask).
    min_move > 0 thins the fixes: a fix closer than min_move metres to the last decoded fix of the same segment
    is left out of the chain (it inherits an arc afterwards, like the other skipped fixes)."""
    net, fx, cands, table = prep["net"], prep["fx"], prep["cands"], prep["table"]
    n = len(fx)
    X, Y, T, SEG, JUMP = fx.x.values, fx.y.values, fx.t.values, fx.segment.values, fx.jump.values
    steps = []          # (fix, candidate arcs, back-pointers or None at a chain start, delta)
    breaks = 0
    delta, prev = None, None
    for k in range(n):
        if k not in cands or JUMP[k]:
            continue                                   # skipped fix: filled from neighbours afterwards
        if (min_move > 0 and prev is not None and SEG[k] == SEG[prev]
                and np.hypot(X[k] - X[prev], Y[k] - Y[prev]) < min_move):
            continue                                   # thinned fix: has not moved far enough from the last one
        A, tA, pA = cands[k]
        q = np.maximum(pA, p_floor)
        e = np.log(q / q.sum())                        # emission: normalised classifier probability
        bp = None
        if prev is not None and SEG[k] == SEG[prev]:
            Ap, tp, _ = cands[prev]
            la, lb = net.len[Ap], net.len[A]
            dt = max(T[k] - T[prev], 1.0)
            d_line = np.hypot(X[k] - X[prev], Y[k] - Y[prev])
            # driving distance: rest of previous arc + shortest path between arcs + part of new arc
            dr = (la * (1 - tp))[:, None] + _lookup(table, net.to[Ap], net.fr[A]) + (lb * tA)[None, :]
            same = Ap[:, None] == A[None, :]
            if same.any():                             # staying on the same arc
                fwd = (tA[None, :] - tp[:, None]) * la[:, None]
                loop = (la * (1 - tp))[:, None] + np.diagonal(_lookup(table, net.to[Ap], net.fr[Ap]))[:, None] \
                    + la[:, None] * tA[None, :]
                dr = np.where(same, np.where(fwd >= -back_tol, np.abs(fwd), loop), dr)
            logT = -np.abs(dr - d_line) / beta
            logT[~(dr <= vmax * dt + slack)] = -np.inf  # cannot be driven in the available time
            score = delta[:, None] + logT
            cand_bp = np.argmax(score, axis=0)
            best = score[cand_bp, np.arange(len(A))]
            if np.isfinite(best).any():
                delta, bp = best + e, cand_bp
            else:
                breaks += 1                            # disconnected: restart the chain at this fix
                delta = e
        else:
            if prev is not None:
                breaks += 1                            # time gap > 60 s (new segment): restart
            delta = e
        steps.append((k, A, bp, delta))
        prev = k
    pred = np.full(n, -1, dtype=np.int64)
    i = len(steps) - 1
    while i >= 0:                                      # backtrack every chain from its own best end state
        j = int(np.argmax(steps[i][3]))
        while True:
            k, A, bp, _ = steps[i]
            pred[k] = A[j]
            i -= 1
            if bp is None:
                break
            j = int(bp[j])
    decoded = pred >= 0
    pred = pd.Series(np.where(decoded, pred, np.nan)).ffill().bfill().fillna(-1).to_numpy().astype(np.int64)
    return pred, breaks, decoded


def reconstruct_route(net, pred, limit=DEFAULTS["limit"]):
    """Turn the per-fix arcs into one connected arc sequence: drop repeats and fill the arcs between two
    consecutive decoded arcs with the shortest path. Returns (route arcs, number of unfillable jumps)."""
    seq = pred[pred >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    if len(seq) == 0:
        return np.array([], dtype=np.int64), 0
    route, holes = [int(seq[0])], 0
    for a, b in zip(seq[:-1], seq[1:]):
        mid = net.path_arcs(int(a), int(b), limit)
        if mid is None:
            holes += 1
        else:
            route.extend(mid)
        route.append(int(b))
    return np.array(route, dtype=np.int64), holes


def snap_to_route(x, y, route, fr, to, nx, ny, chunk=256):
    """Arc of every fix on a given route (arc sequence), with the fixes kept in travel order.

    Each fix gets a position in the route. Positions may never decrease from one fix to the next, and among all
    such assignments the one with the smallest total fix-to-arc distance is chosen (dynamic programming:
    cost_k(m) = dist(fix k, route arc m) + min over m' <= m of cost_{k-1}(m')). This is the same rule
    data_processing/clean.py uses to turn a reference route into per-fix labels, applied here to the PREDICTED
    route, so prediction and label follow one convention about where one arc ends and the next begins.

    The distance matrix is built `chunk` fixes at a time and only the back-pointers are stored (2 bytes per
    fix x route arc), so a 20 000-fix trip on a 2 000-arc route needs about 80 MB.
    """
    route = np.asarray(route, dtype=np.int64)
    n, m = len(x), len(route)
    if m == 0:
        return np.full(n, -1, dtype=np.int64)
    ax, ay, bx, by = nx[fr[route]][None], ny[fr[route]][None], nx[to[route]][None], ny[to[route]][None]
    back = np.empty((n, m), dtype=np.uint16 if m < 65535 else np.int32)
    idx = np.arange(m)
    cost = None
    for s0 in range(0, n, chunk):
        D, _ = pt_seg_dist(x[s0:s0 + chunk, None], y[s0:s0 + chunk, None], ax, ay, bx, by)
        D = D.astype(np.float32)                       # same rounding as the label code
        for i in range(len(D)):
            if cost is None:
                cost, back[0] = D[0].astype(np.float64), idx
                continue
            pm = np.minimum.accumulate(cost)           # best cost among route positions <= m ...
            is_new = np.r_[True, cost[1:] < pm[:-1]]
            back[s0 + i] = np.maximum.accumulate(np.where(is_new, idx, 0))   # ... and where it was reached
            cost = pm + D[i]
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos]


def decode_trip(prep, nx, ny, beta=DEFAULTS["beta"], vmax=DEFAULTS["vmax"], slack=DEFAULTS["slack"],
                back_tol=DEFAULTS["back_tol"], p_floor=DEFAULTS["p_floor"], min_move=DEFAULTS["min_move"],
                limit=DEFAULTS["limit"], snap=True):
    """The whole decoder for one trip: Viterbi -> connected route -> every fix snapped onto that route.
    nx, ny are the projected node coordinates. Returns dict(fix_arc, viterbi_arc, route, breaks, holes)."""
    net, fx = prep["net"], prep["fx"]
    vit, breaks, _ = viterbi_trip(prep, beta=beta, vmax=vmax, slack=slack, back_tol=back_tol, p_floor=p_floor,
                                  min_move=min_move)
    route, holes = reconstruct_route(net, vit, limit=limit)
    fix_arc = snap_to_route(fx.x.values, fx.y.values, route, net.fr, net.to, nx, ny) if snap and len(route) else vit
    return dict(fix_arc=fix_arc, viterbi_arc=vit, route=route, breaks=breaks, holes=holes)
