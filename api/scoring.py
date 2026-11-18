"""Server-side scoring of a predicted route against the reference route. Used only by the API; the reference
route and the labels derived from it never leave the server.

Labels and the "reliable fix" flag come from the same cleaning code that produced the training labels
(data_processing/clean.py, rules C4-C8), so the numbers match the notebooks.
"""
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# make the other code folders importable (config lives in deployment/, the pipeline code in the folders below)
_REPO = Path(__file__).resolve().parent.parent
for _d in ("deployment", "data_processing", "hmm_viterbi", "model_training"):
    if str(_REPO / _d) not in sys.path:
        sys.path.insert(0, str(_REPO / _d))

import config  # noqa: E402

import clean  # noqa: E402


@lru_cache(maxsize=32)
def _truth(trip_dir):
    """(label arc per fix, reliable-fix mask, reference route, arc table) for one trip folder."""
    d = Path(trip_dir)
    with tempfile.TemporaryDirectory() as tmp:
        clean.OUT = tmp
        clean.clean_trip("eval", str(d))
        od = Path(tmp) / "eval" / d.name
        fixes = pd.read_csv(od / "fixes.csv.gz", usecols=["segment", "label_arc", "keep_train"])
        arcs = pd.read_csv(od / "arcs.csv.gz")
    route = np.loadtxt(d / f"{d.name}.route", dtype=int, ndmin=1)
    return fixes, route, arcs


def continuity(pred, segment, fr, to, elig):
    """Share of consecutive fix pairs (same segment) whose predicted arcs are equal or connected within 3 arcs."""
    a, b = pred[:-1], pred[1:]
    ok = (segment[1:] == segment[:-1]) & (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan")
    cont = a == b
    ch = ~cont
    u, v = to[a[ch]], fr[b[ch]]
    n = int(max(fr.max(), to.max())) + 1
    M = csr_matrix((np.ones(int(elig.sum())), (fr[elig], to[elig])), shape=(n, n))
    M.data[:] = 1
    hop1 = np.asarray(M[u, v]).ravel() > 0
    hop2 = np.asarray((M @ M)[u, v]).ravel() > 0
    cont[ch] = (u == v) | hop1 | hop2
    return float(cont.mean())


def score_trip(trip_dir, fix_arc, route):
    """Metrics of a prediction (arc per fix, arc sequence) against the reference. Returns plain floats."""
    fixes, ref, arcs = _truth(str(trip_dir))
    pred = np.asarray(fix_arc, dtype=np.int64)
    lab, keep = fixes.label_arc.values, fixes.keep_train.values
    ok = pred == lab
    fr, to = arcs["from"].values, arcs["to"].values
    length = arcs.length_m.values
    ref_u, pr_u = np.unique(ref), np.unique(np.asarray(route, dtype=np.int64))
    inter = length[np.intersect1d(ref_u, pr_u)].sum()
    P, R = inter / max(length[pr_u].sum(), 1e-9), inter / max(length[ref_u].sum(), 1e-9)
    return dict(
        accuracy=float(ok[keep].mean()) if keep.any() else float(ok.mean()),   # top-1 on reliable fixes
        accuracy_all_fixes=float(ok.mean()),
        route_continuity=continuity(pred, fixes.segment.values, fr, to, arcs.candidate_eligible.values),
        route_precision=float(P), route_recall=float(R), route_f1=float(2 * P * R / (P + R)) if P + R else 0.0,
        n_fixes=int(len(pred)), n_reliable_fixes=int(keep.sum()),
    )
