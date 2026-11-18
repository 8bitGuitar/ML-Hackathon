"""Shared code for the PathSnap model notebooks.

Everything that must be identical across the model notebooks lives here, so that every model is trained and scored
in exactly the same way:
  * loading the candidate feature tables (output of feature_engineering.ipynb) and the cleaned fixes,
  * the GroupKFold(5)-by-trip folds,
  * the metrics (top-1, macro F1, log loss, precision/recall/F1, confusion matrix, route continuity, route F1),
  * the cross-validation loop and the final fit + student_test evaluation,
  * saving results to results/<model>/.
"""
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data_features"
CLEAN_DIR = ROOT / "data_clean"
RAW_DIR = ROOT / "Student folder"
OUT_DIR = ROOT / "results"
SEED = 42
N_FOLDS = 5
TRIP_STRIDE = 100_000  # gid = trip_index * TRIP_STRIDE + fix (largest trip has 19 313 fixes)

FEATURES = json.load(open(FEAT_DIR / "selection_results.json"))["selected"]
LOG_FEATURES = ["dist", "dist_gap", "nbr_dist", "arc_len", "speed"]  # heavy-tailed, log1p before scaling
CAT_FEATURES = ["conn_prev"]                                         # 0/1/2/3 codes, one-hot for non-tree models


# ----------------------------------------------------------------------------------------------- loading
def trips(part):
    return sorted(os.listdir(CLEAN_DIR / part))


def load_partition(part):
    """All candidate rows of a partition (every fix, also the ones not kept for training)."""
    cache = FEAT_DIR / f"cache_{part}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    for i, t in enumerate(trips(part)):
        c = pd.read_csv(FEAT_DIR / part / f"{t}.csv.gz", usecols=["fix", "arc", "y", "keep_train"] + FEATURES)
        c[FEATURES] = c[FEATURES].astype("float32")
        c["tidx"] = np.int16(i)
        frames.append(c)
    df = pd.concat(frames, ignore_index=True)
    df["gid"] = df.tidx.astype(np.int64) * TRIP_STRIDE + df.fix
    df["y"] = df.y.astype(np.int8)
    df.to_pickle(cache)
    return df


@lru_cache(maxsize=None)
def load_fixes(part):
    """One DataFrame per trip with the cleaned fixes (labels, segments, flags)."""
    cols = ["fix", "t", "dt", "x", "y", "segment", "jump", "dwell", "label_arc", "keep_train"]
    return [pd.read_csv(CLEAN_DIR / part / t / "fixes.csv.gz", usecols=cols) for t in trips(part)]


@lru_cache(maxsize=8)
def load_arcs(part, tidx):
    t = trips(part)[tidx]
    a = pd.read_csv(CLEAN_DIR / part / t / "arcs.csv.gz")
    return (a["from"].values, a["to"].values, a.length_m.values.astype(float), a.candidate_eligible.values)


def load_route(part, tidx):
    t = trips(part)[tidx]
    return np.loadtxt(RAW_DIR / part / t / f"{t}.route", dtype=int, ndmin=1)


def matrix(df):
    return df[FEATURES].to_numpy(np.float32)


# ----------------------------------------------------------------------------------------------- preprocessing
def scaled_preprocessor():
    """For scale-sensitive models (linear, kNN, naive Bayes, MLP): log1p on heavy-tailed distances, standardise,
    one-hot the categorical conn_prev. Fitted inside each training fold only (it is part of the model pipeline)."""
    idx = {f: i for i, f in enumerate(FEATURES)}
    log_i = [idx[f] for f in LOG_FEATURES if f in idx]
    cat_i = [idx[f] for f in CAT_FEATURES if f in idx]
    rest = [i for i in range(len(FEATURES)) if i not in log_i + cat_i]
    return ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), log_i),
        ("num", StandardScaler(), rest),
        ("cat", OneHotEncoder(categories=[np.arange(4.0)] * len(cat_i), sparse_output=False), cat_i),
    ])


# ----------------------------------------------------------------------------------------------- metrics
def best_rows(gid, p):
    """Row index of the highest-scoring candidate of every fix."""
    order = np.lexsort((-p, gid))
    g = gid[order]
    return order[np.r_[True, g[1:] != g[:-1]]]


def quick_top1_logloss(gid, y, p):
    b = best_rows(gid, p)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return float(y[b].mean()), float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def fix_predictions(df, p, part):
    """dict trip_index -> predicted arc per fix (-1 if the fix has no candidates)."""
    fxs = load_fixes(part)
    b = best_rows(df.gid.values, p)
    tid, fix, arc = df.tidx.values[b], df.fix.values[b], df.arc.values[b]
    out = {}
    for t in np.unique(df.tidx.values):
        pred = np.full(len(fxs[t]), -1, dtype=np.int64)
        m = tid == t
        pred[fix[m]] = arc[m]
        out[int(t)] = pred
    return out


def continuity(part, tidx, pred, fx=None):
    """Share of consecutive fix pairs (same segment) whose predicted arcs are the same arc or connected within
    3 arcs along the directed network (the reference labels satisfy this for 99.9% of pairs)."""
    fx = load_fixes(part)[tidx] if fx is None else fx
    fr, to, _, elig = load_arcs(part, tidx)
    a, b = pred[:-1], pred[1:]
    ok_pair = (fx.segment.values[1:] == fx.segment.values[:-1]) & (a >= 0) & (b >= 0)
    a, b = a[ok_pair], b[ok_pair]
    if len(a) == 0:
        return np.nan
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


def route_prf(part, tidx, arcs_pred):
    """Length-weighted precision / recall / F1 of the set of predicted arcs against the reference route."""
    _, _, length, _ = load_arcs(part, tidx)
    ref = np.unique(load_route(part, tidx))
    pr = np.unique(arcs_pred[arcs_pred >= 0])
    inter = length[np.intersect1d(ref, pr)].sum()
    P = inter / max(length[pr].sum(), 1e-9)
    R = inter / max(length[ref].sum(), 1e-9)
    return P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)


def trip_metrics(part, tidx, pred, fx=None):
    fx = load_fixes(part)[tidx] if fx is None else fx
    lab, keep = fx.label_arc.values, fx.keep_train.values
    ok = pred == lab
    mf1 = f1_score(lab[keep], pred[keep], average="macro", zero_division=0) if keep.any() else np.nan
    P, R, F = route_prf(part, tidx, pred)
    return dict(n_fix=len(fx), n_kept=int(keep.sum()), correct_kept=int(ok[keep].sum()), correct_all=int(ok.sum()),
                macro_f1=mf1, continuity=continuity(part, tidx, pred, fx), route_p=P, route_r=R, route_f1=F)


def summarise_trips(tm):
    tm = pd.DataFrame(tm)
    return dict(top1=tm.correct_kept.sum() / tm.n_kept.sum(), top1_all=tm.correct_all.sum() / tm.n_fix.sum(),
                macro_f1=tm.macro_f1.mean(), continuity=np.average(tm.continuity, weights=tm.n_fix),
                route_f1=tm.route_f1.mean(), route_p=tm.route_p.mean(), route_r=tm.route_r.mean())


def evaluate(df, p, part, with_probs=True):
    """All metrics for the trips present in df. Fix-level metrics use reliable-label (keep_train) fixes, except
    top1_all (every fix) and continuity / route metrics (the whole predicted sequence)."""
    preds = fix_predictions(df, p, part)
    tm = [dict(tidx=t, **trip_metrics(part, t, pr)) for t, pr in preds.items()]
    res = summarise_trips(tm)
    if with_probs:
        k = df.keep_train.values
        y, pk = df.y.values[k], np.clip(p[k], 1e-7, 1 - 1e-7)
        P, R, F, _ = precision_recall_fscore_support(y, pk >= 0.5, average="binary", zero_division=0)
        res.update(logloss=float(log_loss(y, pk, labels=[0, 1])), precision=float(P), recall=float(R),
                   f1=float(F), cm=confusion_matrix(y, pk >= 0.5, labels=[0, 1]).tolist())
    res["per_trip"] = tm
    return res, preds


# ----------------------------------------------------------------------------------------------- CV + final fit
def make_folds(df):
    """GroupKFold(5) by trip on the reliable-label rows; returns the validation trip indices of each fold."""
    kept = df[df.keep_train]
    return [np.unique(kept.tidx.values[va])
            for _, va in GroupKFold(n_splits=N_FOLDS).split(kept, groups=kept.tidx.values)]


def subsample_fixes(rows, gid, frac, seed):
    """Keep a random `frac` of the fixes (all candidates of a chosen fix stay together)."""
    if frac >= 1:
        return rows
    g = np.unique(gid[rows])
    chosen = np.random.default_rng(seed).choice(g, int(len(g) * frac), replace=False)
    return rows[np.isin(gid[rows], chosen)]


def _fit(model, X, y, val=None):
    if getattr(model, "wants_val", False):
        return model.fit(X, y, val=val)
    return model.fit(X, y)


def run_cv(make_model, df, frac, label="model", verbose=True, train_eval_fixes=20_000, metrics=True):
    """GroupKFold(5) by trip. Each fold: train on a `frac` subsample of the reliable-label fixes of 64 trips,
    predict every candidate of the 16 held-out trips. Returns (fold table, out-of-fold probabilities, models)."""
    X, y, gid = matrix(df), df.y.values, df.gid.values
    keep, tidx = df.keep_train.values, df.tidx.values
    oof = np.full(len(df), np.nan, dtype=np.float32)
    rows, models = [], []
    for k, val_trips in enumerate(make_folds(df)):
        is_val = np.isin(tidx, val_trips)
        tr = subsample_fixes(np.flatnonzero(~is_val & keep), gid, frac, SEED + k)
        va = np.flatnonzero(is_val)
        va_k = np.flatnonzero(is_val & keep)
        model = make_model()
        val_mon = subsample_fixes(va_k, gid, min(1.0, 10_000 / max(1, len(np.unique(gid[va_k])))), SEED)
        t0 = time.time()
        _fit(model, X[tr], y[tr], val=(X[val_mon], y[val_mon], gid[val_mon]))
        fit_s = time.time() - t0
        t0 = time.time()
        oof[va] = model.predict_proba(X[va])[:, 1]
        pred_s = time.time() - t0
        trs = subsample_fixes(tr, gid, min(1.0, train_eval_fixes / len(np.unique(gid[tr]))), SEED)
        tr_top1, tr_ll = quick_top1_logloss(gid[trs], y[trs], model.predict_proba(X[trs])[:, 1])
        va_top1, va_ll = quick_top1_logloss(gid[va_k], y[va_k], oof[va_k])
        row = dict(fold=k + 1, val_trips=len(val_trips), train_rows=len(tr), fit_s=fit_s, predict_s=pred_s,
                   predict_ms_per_1k_fixes=1e6 * pred_s / len(np.unique(gid[va])),
                   train_top1=tr_top1, train_logloss=tr_ll, val_top1=va_top1, val_logloss=va_ll)
        if metrics:
            res, _ = evaluate(df.iloc[va], oof[va], "train")
            row.update({f"val_{m}": res[m] for m in ["macro_f1", "top1_all", "continuity", "route_f1",
                                                      "precision", "recall", "f1"]})
        rows.append(row)
        models.append(model)
        if verbose:
            print(f"[{label}] fold {k + 1}: train top-1 {tr_top1:.4f} | val top-1 {va_top1:.4f} "
                  f"| val log loss {va_ll:.4f} | fit {fit_s:.0f}s", flush=True)
    return pd.DataFrame(rows), oof, models


def fit_and_test(make_model, train_df, test_df, frac):
    """Refit on the reliable-label fixes of all 80 train trips (same `frac` recipe as CV) and score student_test."""
    tr = subsample_fixes(np.flatnonzero(train_df.keep_train.values), train_df.gid.values, frac, SEED)
    X = matrix(train_df)
    model = make_model()
    t0 = time.time()
    _fit(model, X[tr], train_df.y.values[tr])
    fit_s = time.time() - t0
    Xt = matrix(test_df)
    t0 = time.time()
    p = model.predict_proba(Xt)[:, 1]
    pred_s = time.time() - t0
    res, preds = evaluate(test_df, p, "student_test")
    n_fix = sum(len(f) for f in load_fixes("student_test"))
    res.update(fit_s=fit_s, predict_s=pred_s, predict_ms_per_1k_fixes=1e6 * pred_s / n_fix, train_rows=len(tr))
    return model, p, res, preds


# ----------------------------------------------------------------------------------------------- saving
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def save_results(name, family, config, cv, test_res, model=None, oof=None, p_test=None, extra=None):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    cv.to_csv(d / "cv_folds.csv", index=False)
    pd.DataFrame(test_res["per_trip"]).assign(trip=lambda x: [trips("student_test")[t] for t in x.tidx]) \
        .to_csv(d / "student_test_per_trip.csv", index=False)
    summary = dict(name=name, family=family, config=config,
                   cv_mean=cv.drop(columns="fold").mean().to_dict(), cv_std=cv.drop(columns="fold").std().to_dict(),
                   student_test={k: v for k, v in test_res.items() if k != "per_trip"}, seed=SEED, **(extra or {}))
    json.dump(_jsonable(summary), open(d / "results.json", "w"), indent=1)
    if oof is not None:
        np.save(d / "oof_train.npy", oof.astype(np.float32))
    if p_test is not None:
        np.save(d / "proba_student_test.npy", p_test.astype(np.float32))
    if model is not None:
        joblib.dump(model, d / "model.joblib", compress=3)
    return summary


# ----------------------------------------------------------------------------------------------- MLP wrapper
class EpochMLP:
    """sklearn MLPClassifier trained one epoch at a time with partial_fit, so that a real per-epoch history of
    train and validation log loss can be recorded. The validation set is only monitored, never used to stop or
    choose anything (the number of epochs is fixed in advance)."""
    wants_val = True

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=512, epochs=25, seed=SEED):
        self.hidden, self.alpha, self.lr, self.batch_size, self.epochs, self.seed = hidden, alpha, lr, batch_size, epochs, seed

    def fit(self, X, y, val=None):
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(self.seed)
        self.pre = scaled_preprocessor().fit(X)
        Xs = self.pre.transform(X).astype(np.float32)
        self.net = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, batch_size=self.batch_size,
                                 learning_rate_init=self.lr, random_state=self.seed, shuffle=False)
        mon = rng.choice(len(Xs), min(100_000, len(Xs)), replace=False)
        if val is not None:
            Xv, yv, gv = self.pre.transform(val[0]).astype(np.float32), val[1], val[2]
        self.history = []
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xs))
            self.net.partial_fit(Xs[perm], y[perm], classes=[0, 1])
            h = dict(epoch=ep + 1, train_logloss=log_loss(y[mon], self.net.predict_proba(Xs[mon])[:, 1], labels=[0, 1]))
            if val is not None:
                pv = self.net.predict_proba(Xv)[:, 1]
                h["val_top1"], h["val_logloss"] = quick_top1_logloss(gv, yv, pv)
            self.history.append(h)
        return self

    def predict_proba(self, X):
        return self.net.predict_proba(self.pre.transform(X).astype(np.float32))


# ----------------------------------------------------------------------------------------------- plots
BLUE, ORANGE, GREY, RED, GREEN = "#2a78d6", "#eb6834", "#8a8a8a", "#e34948", "#008300"


def plot_cv(cv, title):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    x = cv.fold.values
    axs[0].plot(x, cv.train_top1 * 100, "o-", color=GREY, lw=2, label="train (fitted fixes)")
    axs[0].plot(x, cv.val_top1 * 100, "o-", color=BLUE, lw=2, label="validation (held-out trips)")
    axs[0].set(xlabel="fold", ylabel="top-1 accuracy (%)", title="Top-1 per fold", xticks=x)
    axs[1].plot(x, cv.train_logloss, "o-", color=GREY, lw=2, label="train (fitted fixes)")
    axs[1].plot(x, cv.val_logloss, "o-", color=BLUE, lw=2, label="validation (held-out trips)")
    axs[1].set(xlabel="fold", ylabel="log loss (lower is better)", title="Log loss per fold", xticks=x)
    for a in axs:
        a.grid(alpha=.3)
        a.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_and_calibration(test_df, p, title):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    k = test_df.keep_train.values
    y, pk = test_df.y.values[k], p[k]
    cm = confusion_matrix(y, pk >= 0.5, labels=[0, 1])
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axs[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%} of row)", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black", fontsize=9)
    axs[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["predicted:\nwrong arc", "predicted:\ntrue arc"],
               yticklabels=["actually:\nwrong arc", "actually:\ntrue arc"], title="Confusion matrix (threshold 0.5)")
    fp, mp = calibration_curve(y, np.clip(pk, 0, 1), n_bins=10, strategy="quantile")
    axs[1].plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfectly calibrated")
    axs[1].plot(mp, fp, "o-", color=ORANGE, lw=2, label="model")
    axs[1].set(xlabel="predicted probability", ylabel="observed share of true arcs", title="Calibration")
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + " — student_test, candidate level")
    fig.tight_layout()
    return fig
