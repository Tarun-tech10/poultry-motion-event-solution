#!/usr/bin/env python3
"""
Hen-House Motion Event Prediction  --  self-contained solution.

Usage:  python3 solution.py <public_dir> <submission_out.csv>

Approach
--------
1. Decode each 16x30 motion strip and use `zone_cards` to undo the row-local zone
   aliasing, mapping every row onto a canonical 4x4 spatial grid (cell = row*4+col).
   All downstream features are built in that grid frame and are invariant to the
   generator's per-row dihedral transform, so nothing depends on fixed zone meanings.
2. Per-cell feature set: multi-scale recency levels / ranks / shares, trends and
   slopes, neighbourhood aggregates on the lattice, optical-flow advection forecasts
   (translation estimated by shift cross-correlation, then bilinear-advected to each
   horizon), damped per-cell AR forecasts, leader/alternation history, and
   peak-trajectory extrapolation.
3. Zone: an ensemble of gradient-boosted per-cell relevance models (one per horizon),
   averaged in log space and renormalised into a 16-way distribution.
   Tier: a 4-class model conditioned on the winning cell, marginalised over the
   predicted zone distribution.
   Direction: DC is emitted -- measured on training data, no compass direction has
   enough mass to beat DC under the grader's asymmetric direction rule.
4. Decode: instead of taking argmaxes, enumerate all 16^3 zone triples and maximise
   the grader's own expected row score (Manhattan-1 partial credit, the zone-bigram
   TransitionScore and the exact-event bonus), with a mild stickiness prior and a
   risk-aversion term that lifts the bottom-20% tail.
"""
import sys, os, json
import numpy as np
import pandas as pd

SEED = 7
np.random.seed(SEED)

ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
IDX = np.full(256, 0, np.int16)
for _i, _c in enumerate(ALPHA):
    IDX[ord(_c)] = _i
ALLD = ["DN", "DNE", "DE", "DSE", "DS", "DSW", "DW", "DNW", "DC"]

CR = np.array([[r, c] for r in range(4) for c in range(4)], np.float32)
MANH = np.abs(CR[:, None] - CR[None, :]).sum(-1)
A1 = (MANH == 1).astype(np.float32)
A2 = (MANH == 2).astype(np.float32)
NNB = A1.sum(1)

# tuned on source-proxy grouped CV against the published metric
DEC = dict(lam=1.5, T=1.15, Tm=1.15, beta=0.0, beta_m=0.0)          # with the GNN blended in
DEC_GBM_ONLY = dict(lam=1.5, T=0.9, Tm=1.0, beta=0.0, beta_m=0.0)   # fallback if torch is absent
GNN_W, GNN_WM = 0.6, 0.5        # zone / tier blend weight on the GNN side
GNN_SEEDS = [1, 2, 3, 4]
GNN_STEPS = 520                 # matched to CV (40 epochs x 13 steps on 4/5 of the data)


# ----------------------------------------------------------------- data loading
def load(path, with_target=True):
    df = pd.read_csv(path)
    n = len(df)
    strips = np.zeros((n, 16, 30), np.float32)
    rc = np.zeros((n, 16, 2), np.int64)
    for i, (s, zc) in enumerate(zip(df["past_motion_strip"].values, df["zone_cards"].values)):
        strips[i] = IDX[np.frombuffer(str(s).encode(), np.uint8)].reshape(16, 30)
        cards = sorted(json.loads(zc), key=lambda z: z["zone"])
        for j, z in enumerate(cards):
            rc[i, int(z["zone"][1:]) - 1] = (z["grid_row"], z["grid_col"])
    out = {"id": df["id"].values, "strip": strips, "rc": rc}
    if with_target and "target_future_ledger" in df.columns:
        Z = np.zeros((n, 3), np.int64); M = np.zeros((n, 3), np.int64)
        for i, t in enumerate(df["target_future_ledger"].values):
            for k, tok in enumerate(str(t).split()):
                p = tok.split(":")
                Z[i, k] = int(p[1][1:]) - 1
                M[i, k] = int(p[3][1:])
        out["zone"] = Z; out["tier"] = M
    return out


def to_cells(strip, rc):
    """(N,16,30) row-local -> (N,16,30) indexed by canonical grid cell r*4+c."""
    n = strip.shape[0]
    g = np.zeros((n, 16, 30), np.float32)
    cell = rc[:, :, 0] * 4 + rc[:, :, 1]
    np.put_along_axis(g, cell[:, :, None].repeat(30, 2), strip, 1)
    return g, cell


# ------------------------------------------------------------------- features
SH = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], np.float32)


def _advect(F, v):
    n = F.shape[0]
    rr, cc = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    sr = np.clip(rr[None] - v[:, 0][:, None, None], 0, 3)
    sc = np.clip(cc[None] - v[:, 1][:, None, None], 0, 3)
    r0 = np.floor(sr).astype(int); c0 = np.floor(sc).astype(int)
    r1 = np.minimum(r0 + 1, 3); c1 = np.minimum(c0 + 1, 3)
    fr = sr - r0; fc = sc - c0
    idx = np.arange(n)[:, None, None]
    return (F[idx, r0, c0] * (1 - fr) * (1 - fc) + F[idx, r1, c0] * fr * (1 - fc)
            + F[idx, r0, c1] * (1 - fr) * fc + F[idx, r1, c1] * fr * fc)


def _flow(G4, lag):
    n, _, _, T = G4.shape
    a = G4[:, :, :, :T - lag]; b = G4[:, :, :, lag:]
    a = a - a.mean((1, 2), keepdims=True); b = b - b.mean((1, 2), keepdims=True)
    sc = np.zeros((n, 9), np.float32)
    for i, (dr, dc) in enumerate(SH):
        dr = int(dr); dc = int(dc)
        r0, r1 = max(0, -dr), 4 - max(0, dr)
        c0, c1 = max(0, -dc), 4 - max(0, dc)
        aa = a[:, r0:r1, c0:c1]; bb = b[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        sc[:, i] = (aa * bb).sum((1, 2, 3)) / (aa.shape[1] * aa.shape[2])
    sc = sc - sc.min(1, keepdims=True)
    w = sc / (sc.sum(1, keepdims=True) + 1e-6)
    return (w @ SH / lag).astype(np.float32), (w.max(1) - w.mean(1)).astype(np.float32)


def zone_features(G):
    """G:(N,16,30) cell-indexed -> (N,16,F)"""
    n = G.shape[0]
    G4 = G.reshape(n, 4, 4, 30)
    gm = G.mean((1, 2))[:, None] + 1e-6
    f = []
    def add(a): f.append(np.asarray(a, np.float32))

    tot = G.sum(1) + 1e-6
    W = {}
    for w in [1, 2, 3, 5, 8, 12, 20, 30]:
        m = G[:, :, -w:].mean(2); W[w] = m
        add(m / gm); add(m - G[:, :, -w:].mean((1, 2))[:, None])
        add((-m).argsort(1).argsort(1))
        add(m / (m.max(1, keepdims=True) + 1e-6))
    for w in [3, 8, 30]:
        add(G[:, :, -w:].sum(2) / (tot[:, -w:].sum(1)[:, None] + 1e-6))
    add((W[3] - G[:, :, -6:-3].mean(2)) / gm)
    add((W[5] - G[:, :, -10:-5].mean(2)) / gm)
    add((W[8] - G[:, :, -16:-8].mean(2)) / gm)
    add((W[12] - G[:, :, :12].mean(2)) / gm)
    t = np.arange(30, dtype=np.float32) - 14.5
    add((G * t).sum(2) / (t ** 2).sum() / gm)
    t10 = np.arange(10, dtype=np.float32) - 4.5
    add((G[:, :, -10:] * t10).sum(2) / (t10 ** 2).sum() / gm)
    add(G.max(2) / gm); add(G.std(2) / gm); add(G.argmax(2).astype(np.float32))
    add((G[:, :, -10:] > G.mean(2, keepdims=True)).mean(2))
    add((G[:, :, -10:] == 0).mean(2))
    d1 = np.diff(G, axis=2)
    add(np.abs(d1[:, :, -10:]).mean(2) / gm); add(d1[:, :, -10:].mean(2) / gm)
    for w in [3, 8]:
        m = W[w]
        n1 = m @ A1.T / NNB[None, :]
        n2 = m @ A2.T / (A2.sum(1)[None, :] + 1e-6)
        add(n1 / gm); add(n2 / gm); add(m / (n1 + 1e-6))
        add((m[:, None, :] * A1[None]).max(2) / gm)
    add(((W[3] - G[:, :, -6:-3].mean(2)) @ A1.T) / NNB[None, :] / gm)
    add(np.broadcast_to(NNB[None, :], (n, 16)))
    add(np.broadcast_to(np.abs(CR - 1.5).sum(1)[None, :], (n, 16)))
    add(np.broadcast_to((NNB == 2).astype(np.float32)[None, :], (n, 16)))
    wts = G + 1e-6
    cen = np.einsum("nct,cd->ntd", wts, CR) / wts.sum(1)[:, :, None]
    tt = np.arange(30, dtype=np.float32) - 14.5
    vel = (cen * tt[None, :, None]).sum(1) / (tt ** 2).sum()
    t10b = np.arange(10, dtype=np.float32) - 4.5
    vel10 = (cen[:, -10:] * t10b[None, :, None]).sum(1) / (t10b ** 2).sum()
    last = cen[:, -3:].mean(1)
    rel = CR[None] - last[:, None, :]
    dist = np.linalg.norm(rel, axis=2) + 1e-6
    add(dist)
    for v in (vel, vel10):
        add((rel * v[:, None, :]).sum(2) / dist)
        add(np.broadcast_to(np.linalg.norm(v, axis=1)[:, None], (n, 16)))
        for hstep in (6, 12, 18):
            add(np.linalg.norm(CR[None] - (last + v * hstep)[:, None, :], axis=2))
    srt = np.sort(W[3], 1)[:, ::-1]
    p3 = W[3] / W[3].sum(1, keepdims=True)
    for v in [G.mean((1, 2)), G.std((1, 2)),
              G[:, :, -10:].mean((1, 2)) - G[:, :, :10].mean((1, 2)),
              srt[:, 0] / (W[3].mean(1) + 1e-6), srt[:, 1] / (srt[:, 0] + 1e-6),
              srt[:, 2] / (srt[:, 0] + 1e-6),
              -(p3 * np.log(p3 + 1e-9)).sum(1), (G == 0).mean((1, 2)),
              G.max((1, 2)), tot.std(1) / (tot.mean(1) + 1e-6)]:
        add(np.broadcast_to(np.asarray(v, np.float32)[:, None], (n, 16)))
    # --- feats2 block: raw recent values, EWMAs, rank dynamics, leader history ---
    for j in range(1, 13):
        add(G[:, :, -j] / gm)
    for tau in [1.0, 2.0, 4.0, 8.0]:
        w = np.exp(-np.arange(29, -1, -1) / tau); w /= w.sum()
        e = (G * w[None, None, :]).sum(2)
        add(e / gm); add((-e).argsort(1).argsort(1))
    rk = lambda a: (-a).argsort(1).argsort(1).astype(np.float32)
    r3 = rk(G[:, :, -3:].mean(2)); r8 = rk(G[:, :, -11:-3].mean(2)); r15 = rk(G[:, :, -30:-11].mean(2))
    add(r3 - r8); add(r8 - r15); add((r3 == 0).astype(np.float32)); add((r8 == 0).astype(np.float32))
    lead = (G == G.max(1, keepdims=True)).astype(np.float32)
    add(lead[:, :, -5:].mean(2)); add(lead[:, :, -15:].mean(2)); add(lead.mean(2))
    thr = np.percentile(G, 75, axis=(1, 2))[:, None, None]
    add((G[:, :, -10:] > thr).mean(2)); add((G > thr).mean(2))
    add(G[:, :, -10:].std(2) / (G[:, :, -10:].mean(2) + 1e-6))
    add((G * np.arange(30, dtype=np.float32)[None, None, :]).sum(2) / (G.sum(2) + 1e-6))
    tz = (tot - tot.mean(1, keepdims=True)) / (tot.std(1, keepdims=True) + 1e-6)
    gz = (G - G.mean(2, keepdims=True)) / (G.std(2, keepdims=True) + 1e-6)
    add((gz * tz[:, None, :]).mean(2))
    # --- feats3 block: flow advection, AR forecasts, alternation, peak track ---
    for lag in [1, 3]:
        v, conf = _flow(G4, lag)
        add(np.broadcast_to(conf[:, None], (n, 16)))
        add(np.broadcast_to(np.linalg.norm(v, axis=1)[:, None], (n, 16)))
        base = G4[:, :, :, -4:].mean(3)
        for off in (3.5, 9.5, 15.5):
            pred = _advect(base, v * off).reshape(n, 16)
            add(pred / gm); add((-pred).argsort(1).argsort(1))
        m3c = G[:, :, -3:].mean(2)
        cen2 = (m3c[:, :, None] * CR[None]).sum(1) / (m3c.sum(1)[:, None] + 1e-6)
        rel2 = CR[None] - cen2[:, None, :]
        add((rel2 * v[:, None, :]).sum(2) / (np.linalg.norm(rel2, axis=2) + 1e-6))
    m3 = G[:, :, -3:].mean(2); m8 = G[:, :, -8:].mean(2); m20 = G[:, :, -20:].mean(2)
    trd = m3 - G[:, :, -6:-3].mean(2)
    for damp in (1.0, 2.0, 3.0):
        pr = 0.6 * (m3 + trd * damp * 0.5) + 0.4 * m8
        add(pr / gm); add((-pr).argsort(1).argsort(1))
    add((m3 - m20) / gm)
    li = G.argmax(1)
    add(np.broadcast_to((np.diff(li, axis=1) != 0).mean(1)[:, None], (n, 16)))
    idxT = np.arange(30)[None, None, :]
    lastlead = np.where(G == G.max(1, keepdims=True), idxT, -1).max(2)
    add(29 - lastlead); add((lastlead >= 0).astype(np.float32))
    cur = li[:, -1]
    ls = G[np.arange(n), cur]
    lz = (ls - ls.mean(1, keepdims=True)) / (ls.std(1, keepdims=True) + 1e-6)
    add((gz * lz[:, None, :]).mean(2))
    add((G[:, :, -6:].mean(2) - ls[:, -6:].mean(1)[:, None]) / gm)
    add(np.broadcast_to((np.sort(m3, 1)[:, -1] - np.sort(m3, 1)[:, -2])[:, None] / gm, (n, 16)))
    add(np.abs(CR[None] - CR[cur][:, None, :]).sum(2))
    pk = CR[li]
    tp = np.arange(30, dtype=np.float32) - 14.5
    pv = ((pk - pk.mean(1, keepdims=True)) * tp[None, :, None]).sum(1) / (tp ** 2).sum()
    plast = pk[:, -5:].mean(1)
    for off in (3.5, 9.5, 15.5):
        add(np.linalg.norm(CR[None] - (plast + pv * off)[:, None, :], axis=2))
    add(np.broadcast_to(np.linalg.norm(pv, axis=1)[:, None], (n, 16)))
    return np.stack(f, -1)


def row_features(G):
    n = G.shape[0]
    m3 = G[:, :, -3:].mean(2); srt = np.sort(m3, 1)[:, ::-1]
    tot = G.sum(1)
    out = []
    def add(a): out.append(np.asarray(a, np.float32))
    for w in [1, 2, 3, 5, 8, 15, 30]:
        add(G[:, :, -w:].mean((1, 2))); add(G[:, :, -w:].max((1, 2)))
        add(np.sort(G[:, :, -w:].mean(2), 1)[:, -1])
        add(np.sort(G[:, :, -w:].mean(2), 1)[:, -3:].mean(1))
    add(G[:, :, -10:].mean((1, 2)) - G[:, :, :10].mean((1, 2)))
    add(G[:, :, -5:].mean((1, 2)) / (G[:, :, -20:-5].mean((1, 2)) + 1e-6))
    add(tot.std(1) / (tot.mean(1) + 1e-6))
    add(np.diff(tot, axis=1).std(1) / (tot.mean(1) + 1e-6))
    add(srt[:, 0] / (m3.mean(1) + 1e-6)); add(srt[:, 1] / (srt[:, 0] + 1e-6))
    add((G == 0).mean((1, 2))); add(G.std((1, 2)))
    add(np.percentile(G, 90, axis=(1, 2))); add(np.percentile(G[:, :, -10:], 99, axis=(1, 2)))
    top = np.argsort(-m3, 1)[:, :3]
    tg = np.take_along_axis(G, top[:, :, None], 1)
    add(tg[:, :, -5:].mean((1, 2)) - tg[:, :, -15:-5].mean((1, 2)))
    add(tg[:, :, -3:].mean((1, 2)))
    return np.stack(out, 1)


# --------------------------------------------------------------------- models
ZCFG = [
    dict(learning_rate=0.030, num_leaves=63, min_data_in_leaf=80, feature_fraction=0.45,
         bagging_fraction=0.80, lambda_l2=8.0, seed=1, rounds=320),
    dict(learning_rate=0.035, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.65,
         bagging_fraction=0.70, lambda_l2=3.0, seed=2, rounds=300),
    dict(learning_rate=0.025, num_leaves=127, min_data_in_leaf=120, feature_fraction=0.35,
         bagging_fraction=0.85, lambda_l2=15.0, seed=3, rounds=340),
    dict(learning_rate=0.040, num_leaves=48, min_data_in_leaf=60, feature_fraction=0.55,
         bagging_fraction=0.75, lambda_l2=6.0, seed=4, rounds=260),
]
TOPC = 6

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def _fit_binary(X, y, cfg, threads):
    if HAVE_LGB:
        p = dict(objective="binary", verbose=-1, num_threads=threads, max_bin=127, bagging_freq=1,
                 learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                 min_data_in_leaf=cfg["min_data_in_leaf"], feature_fraction=cfg["feature_fraction"],
                 bagging_fraction=cfg["bagging_fraction"], lambda_l2=cfg["lambda_l2"], seed=cfg["seed"])
        m = lgb.train(p, lgb.Dataset(X, y), cfg["rounds"])
        return lambda Q: m.predict(Q)
    m = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_iter=cfg["rounds"], max_leaf_nodes=cfg["num_leaves"],
        min_samples_leaf=cfg["min_data_in_leaf"], l2_regularization=cfg["lambda_l2"],
        max_bins=127, random_state=cfg["seed"], early_stopping=False).fit(X, y)
    return lambda Q: m.predict_proba(Q)[:, 1]


def _fit_multi(X, y, threads, rounds=420):
    if HAVE_LGB:
        p = dict(objective="multiclass", num_class=4, learning_rate=0.035, num_leaves=31,
                 min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=5.0, verbose=-1, max_bin=127, num_threads=threads, seed=SEED)
        m = lgb.train(p, lgb.Dataset(X, y), rounds)
        return lambda Q: m.predict(Q)
    m = HistGradientBoostingClassifier(learning_rate=0.035, max_iter=rounds, max_leaf_nodes=31,
                                       min_samples_leaf=60, l2_regularization=5.0, max_bins=127,
                                       random_state=SEED, early_stopping=False).fit(X, y)
    def pred(Q):
        p = m.predict_proba(Q)
        full = np.zeros((len(Q), 4)); full[:, m.classes_.astype(int)] = p
        return full
    return pred


RANK_W, RANK_T = 0.35, 0.75
RCFG = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=80, feature_fraction=0.45,
            bagging_fraction=0.8, lambda_l2=8.0, seed=1, rounds=320)


def fit_zone(XZtr, ZCtr, XZte, threads=8):
    ntr, _, F = XZtr.shape; nte = XZte.shape[0]
    Ftr = XZtr.reshape(-1, F); Fte = XZte.reshape(-1, F)
    out = np.zeros((nte, 3, 16), np.float32)
    for k in range(3):
        y = np.zeros((ntr, 16), np.float32); y[np.arange(ntr), ZCtr[:, k]] = 1
        acc = np.zeros((nte, 16))
        for cfg in ZCFG:
            p = _fit_binary(Ftr, y.reshape(-1), cfg, threads)(Fte).reshape(nte, 16)
            acc += np.log(np.clip(p, 1e-7, 1))
        acc /= len(ZCFG)
        if HAVE_LGB:                       # decorrelated lambdarank member
            rp = dict(objective="lambdarank", lambdarank_truncation_level=4, verbose=-1,
                      num_threads=threads, max_bin=127, bagging_freq=1,
                      learning_rate=RCFG["learning_rate"], num_leaves=RCFG["num_leaves"],
                      min_data_in_leaf=RCFG["min_data_in_leaf"], feature_fraction=RCFG["feature_fraction"],
                      bagging_fraction=RCFG["bagging_fraction"], lambda_l2=RCFG["lambda_l2"], seed=RCFG["seed"])
            rm = lgb.train(rp, lgb.Dataset(Ftr, y.reshape(-1), group=np.full(ntr, 16)), RCFG["rounds"])
            rs = rm.predict(Fte).reshape(nte, 16) / RANK_T
            rs = rs - rs.max(1, keepdims=True)
            acc = (1 - RANK_W) * acc + RANK_W * (rs - np.log(np.exp(rs).sum(1, keepdims=True)))
        e = np.exp(acc - acc.max(1, keepdims=True))
        out[:, k] = (e / e.sum(1, keepdims=True)).astype(np.float32)
    return out


def fit_tier(XZtr, XRtr, Mtr, ZCtr, XZte, XRte, pz, threads=8):
    nte = XZte.shape[0]; ntr = XZtr.shape[0]
    out = np.zeros((nte, 3, 4), np.float32)
    cand = np.argsort(-pz, axis=2)[:, :, :TOPC]
    for k in range(3):
        Xtr = np.concatenate([XRtr, XZtr[np.arange(ntr), ZCtr[:, k]]], 1)
        pred = _fit_multi(Xtr, Mtr[:, k], threads)
        num = np.zeros((nte, 4)); den = np.zeros((nte, 1))
        for j in range(TOPC):
            c = cand[:, k, j]
            w = pz[np.arange(nte), k, c][:, None]
            num += w * pred(np.concatenate([XRte, XZte[np.arange(nte), c]], 1))
            den += w
        out[:, k] = (num / np.maximum(den, 1e-9)).astype(np.float32)
    return out


# ------------------------------------------------------------------- lattice GNN
# Per-cell temporal encoder + message passing on the 4x4 lattice, D4-augmented and
# D4-TTA'd. Deliberately small (no Conv3d) so it trains on CPU in ~90 s per seed.
try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as TF
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _d4perm(t):
    g = np.arange(16).reshape(4, 4)
    if t & 4:
        g = g.T
    g = np.rot90(g, t & 3)
    return g.reshape(-1).copy()


PERM = np.stack([_d4perm(t) for t in range(8)])
INV = np.stack([np.argsort(p) for p in PERM])


def _build_net(dm=96):
    A = (MANH == 1).astype(np.float32)
    A = A / A.sum(1, keepdims=True)
    At = torch.tensor(A)
    node = np.stack([np.abs(CR - 1.5).sum(1) / 3.0, NNB / 4.0], 1).astype(np.float32)
    NODEt = torch.tensor(node)

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.tenc = tnn.Sequential(
                tnn.Conv1d(2, 48, 5, padding=2), tnn.GELU(),
                tnn.Conv1d(48, 64, 5, padding=2), tnn.GELU(),
                tnn.Conv1d(64, 64, 3, padding=2, dilation=2), tnn.GELU())
            s.proj = tnn.Linear(64 * 3 + 2, dm)
            s.mp = tnn.ModuleList([tnn.Linear(dm * 3, dm) for _ in range(3)])
            s.ln = tnn.ModuleList([tnn.LayerNorm(dm) for _ in range(3)])
            s.zhead = tnn.Linear(dm, 3)
            s.mhead = tnn.Sequential(tnn.Linear(dm * 2, 128), tnn.GELU(), tnn.Linear(128, 12))

        def forward(s, x):
            B = x.shape[0]
            h = s.tenc(x.permute(0, 2, 1, 3).reshape(B * 16, 2, 30))
            h = torch.cat([h[:, :, -1], h[:, :, -6:].mean(2), h.mean(2)], 1).reshape(B, 16, -1)
            h = s.proj(torch.cat([h, NODEt.expand(B, 16, 2)], 2))
            for lin, ln in zip(s.mp, s.ln):
                nb = torch.einsum("ij,bjd->bid", At, h)
                g = h.mean(1, keepdim=True).expand_as(h)
                h = ln(h + TF.gelu(lin(torch.cat([h, nb, g], 2))))
            pooled = torch.cat([h.mean(1), h.max(1).values], 1)
            return s.zhead(h).permute(0, 2, 1), s.mhead(pooled).reshape(B, 3, 4)
    return Net


def gnn_input(G):
    rm = G.mean((1, 2), keepdims=True) + 1e-6
    return np.stack([G / rm, np.log1p(G) - np.log1p(G.mean((1, 2), keepdims=True))], 1).astype(np.float32)


def fit_gnn(Gtr, ZCtr, Mtr, Gte, threads):
    torch.set_num_threads(threads)
    Xtr = torch.tensor(gnn_input(Gtr)); Xte = torch.tensor(gnn_input(Gte))
    ZCt = torch.tensor(ZCtr); Mt = torch.tensor(Mtr.astype(np.int64))
    ntr = Xtr.shape[0]; nte = Xte.shape[0]
    Net = _build_net()
    zacc = np.zeros((nte, 3, 16)); macc = np.zeros((nte, 3, 4))
    for sd in GNN_SEEDS:
        torch.manual_seed(sd); rng = np.random.RandomState(sd)
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=GNN_STEPS, pct_start=0.25)
        BS = 256; st = 0
        while st < GNN_STEPS:
            perm = rng.permutation(ntr)
            for i in range(0, ntr - BS + 1, BS):
                if st >= GNN_STEPS:
                    break
                b = perm[i:i + BS]; t = rng.randint(8)
                xb = Xtr[b][:, :, PERM[t], :]
                zb = torch.tensor(INV[t])[ZCt[b]]
                zl, ml = net(xb)
                loss = (TF.cross_entropy(zl.reshape(-1, 16), zb.reshape(-1), label_smoothing=0.05)
                        + 0.4 * TF.cross_entropy(ml.reshape(-1, 4), Mt[b].reshape(-1), label_smoothing=0.05))
                opt.zero_grad(); loss.backward()
                tnn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if st < GNN_STEPS - 1:
                    sch.step()
                st += 1
        net.eval()
        pz = np.zeros((nte, 3, 16)); pm = np.zeros((nte, 3, 4))
        with torch.no_grad():
            for t in range(8):
                for i in range(0, nte, 512):
                    zl, ml = net(Xte[i:i + 512][:, :, PERM[t], :])
                    pz[i:i + 512] += TF.softmax(zl.float(), -1).numpy()[:, :, PERM[t]] / 8
                    pm[i:i + 512] += TF.softmax(ml.float(), -1).numpy() / 8
        zacc += np.log(np.clip(pz, 1e-7, 1)); macc += pm
        print("  gnn seed %d done" % sd, flush=True)
    zacc /= len(GNN_SEEDS)
    e = np.exp(zacc - zacc.max(2, keepdims=True))
    return (e / e.sum(2, keepdims=True)).astype(np.float32), (macc / len(GNN_SEEDS)).astype(np.float32)


# --------------------------------------------------------------------- decode
ADJ = (MANH == 1).astype(np.float32)
TADJ = np.zeros((4, 4), np.float32)
for a in range(4):
    for b in range(4):
        TADJ[a, b] = 1.0 if a == b else (0.1 if abs(a - b) == 1 else 0.0)
TRIP = np.array([(a, b, c) for a in range(16) for b in range(16) for c in range(16)])


def decode(pz, pm, lam, beta, beta_m, w_trans=0.18):
    n = pz.shape[0]
    Zs = pz + 0.1 * (pz @ ADJ.T)
    Zn = pz + (pz @ ADJ.T)
    Ts = pm @ TADJ.T
    Tn = pm @ (TADJ > 0).astype(np.float32).T
    msel = (Ts + beta_m * Tn).argmax(2)
    ar = np.arange(n)[:, None]; hh = np.arange(3)[None, :]
    pm_s = pm[ar, hh, msel]
    p_dc = 1.0                                       # direction fixed to DC
    base = (0.64 / 3) * (0.52 * Zs + 0.10 * pz * p_dc * pm_s[:, :, None]) + beta * Zn
    a0 = base[:, 0][:, TRIP[:, 0]]; a1 = base[:, 1][:, TRIP[:, 1]]; a2 = base[:, 2][:, TRIP[:, 2]]
    p0 = pz[:, 0][:, TRIP[:, 0]]; p1 = pz[:, 1][:, TRIP[:, 1]]; p2 = pz[:, 2][:, TRIP[:, 2]]
    eq01 = (TRIP[:, 0] == TRIP[:, 1])[None, :]; eq12 = (TRIP[:, 1] == TRIP[:, 2])[None, :]
    obj = a0 + a1 + a2 + (w_trans / 2.0) * (p0 * p1 * (1 + lam * eq01) + p1 * p2 * (1 + lam * eq12))
    return TRIP[obj.argmax(1)], msel


# ----------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    pub, outp = sys.argv[1], sys.argv[2]
    threads = max(1, min(16, (os.cpu_count() or 4)))

    tr = load(os.path.join(pub, "train.csv"), True)
    te = load(os.path.join(pub, "test.csv"), False)
    Gtr, cell_tr = to_cells(tr["strip"], tr["rc"])
    Gte, cell_te = to_cells(te["strip"], te["rc"])
    ZCtr = np.take_along_axis(cell_tr, tr["zone"], 1)

    XZtr = zone_features(Gtr); XZte = zone_features(Gte)
    XRtr = row_features(Gtr);  XRte = row_features(Gte)

    pz = fit_zone(XZtr, ZCtr, XZte, threads)
    pm = fit_tier(XZtr, XRtr, tr["tier"], ZCtr, XZte, XRte, pz, threads)
    print("gbm done", flush=True)

    dec = DEC
    if HAVE_TORCH:
        gz, gm = fit_gnn(Gtr, ZCtr, tr["tier"], Gte, threads)
        lo = (1 - GNN_W) * np.log(np.clip(pz, 1e-7, 1)) + GNN_W * np.log(np.clip(gz, 1e-7, 1))
        e = np.exp(lo - lo.max(2, keepdims=True))
        pz = (e / e.sum(2, keepdims=True)).astype(np.float32)
        pm = ((1 - GNN_WM) * pm + GNN_WM * gm).astype(np.float32)
    else:
        dec = DEC_GBM_ONLY
        print("torch unavailable -> GBM-only fallback", flush=True)

    p = pz ** (1.0 / dec["T"]); p /= p.sum(2, keepdims=True)
    q = pm ** (1.0 / dec["Tm"]); q /= q.sum(2, keepdims=True)
    cells, tiers = decode(p, q, dec["lam"], dec["beta"], dec["beta_m"])

    # cell -> row-local alias
    nte = len(te["id"])
    cell2alias = np.zeros((nte, 16), np.int64)
    np.put_along_axis(cell2alias, cell_te, np.arange(16)[None, :].repeat(nte, 0), 1)
    alias = np.take_along_axis(cell2alias, cells, 1)

    led = ["%s %s %s" % tuple(
        "H%d:Z%02d:DC:M%d" % (k, alias[i, k] + 1, tiers[i, k]) for k in range(3))
        for i in range(nte)]
    pd.DataFrame({"id": te["id"], "predicted_future_ledger": led}).to_csv(outp, index=False)
    print("wrote %s  rows=%d" % (outp, nte))


if __name__ == "__main__":
    main()
