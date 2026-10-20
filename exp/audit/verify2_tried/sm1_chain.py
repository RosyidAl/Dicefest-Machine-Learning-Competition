"""Independent verification of the small-meter-count (bld_dim_12 <= 4) scaling claim on the holdout chain.

Uses only exp/audit/chain_H.npz. Groups: M<=3, M==4 (and M==5 as a control).
Checks:
  1. bias of corrected / teacher predictions on H and of the P-side fitted blend (P-OOF), per group and metric
  2. P -> H: factor f = (y.p)/(p.p) fitted on P (teacher level), applied to H corrected with shrink s
  3. H internal: repeated random 2-fold cross-fitting on H (fit on one half's corrected preds, apply to other half)
  4. H -> P (reverse direction): fit on H corrected, apply to the P-side blend, scored on P
  5. uncertainty: bootstrap of H rows (fixed f), bootstrap of P group rows (f noise), public-size subsets
All effects reported as change of the TOTAL score (= per-metric NRMSE change / 5).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_tried/sm1_chain.py
"""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ["gas", "eui", "elec"]
Z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = Z["P"], Z["H"], Z["M"]
MP, MH = M[P], M[H]
GROUPS = {"le3": lambda m: m <= 3, "eq4": lambda m: m == 4}
CTRL = {"le3": lambda m: m <= 3, "eq4": lambda m: m == 4, "eq5": lambda m: m == 5, "6to7": lambda m: (m >= 6) & (m <= 7)}

D = {}
for m in MET:
    yH, tH, cH = Z[f"{m}_y"].astype(float), Z[f"{m}_teacher"].astype(float), Z[f"{m}_corrected"].astype(float)
    pP = Z[f"{m}_pP"].astype(float)
    yP = pP + Z[f"{m}_rP"].astype(float)
    D[m] = dict(yH=yH, tH=tH, cH=cH, pP=pP, yP=yP)


def nrmse(y, p, m, w=None):
    if w is None:
        return np.sqrt(np.mean((y - p) ** 2)) / SIG[m]
    return np.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)) / SIG[m]


def lsf(y, p):
    return float(y @ p / (p @ p))


def stats(y, p):
    r = y - p
    n = len(r)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return dict(n=n, rel=float(r.sum() / p.sum()), f=lsf(y, p), t=float(t), frac_pos=float(np.mean(r > 0)),
                med_ratio=float(np.median(y / p)))


print("== 1. bias per group (rel = sum(y-p)/sum(p); f = LS scale; t of mean residual; frac y>p; median y/p)")
for m in MET:
    d = D[m]
    for g, fn in CTRL.items():
        sH, sP = fn(MH), fn(MP)
        a = stats(d["yH"][sH], d["cH"][sH])
        b = stats(d["yH"][sH], d["tH"][sH])
        c = stats(d["yP"][sP], d["pP"][sP])
        print(f"{m:4s} {g:4s} H-corr n={a['n']:5d} rel {a['rel']:+.4f} f {a['f']:.4f} t {a['t']:+5.1f} pos {a['frac_pos']:.2f} "
              f"med {a['med_ratio']:.3f} | H-teach rel {b['rel']:+.4f} f {b['f']:.4f} t {b['t']:+5.1f} | "
              f"P n={c['n']:5d} rel {c['rel']:+.4f} f {c['f']:.4f} t {c['t']:+5.1f} pos {c['frac_pos']:.2f} med {c['med_ratio']:.3f}")


def apply_rule(pred, Mv, fac, s):
    out = pred.copy()
    for g, fn in GROUPS.items():
        sel = fn(Mv)
        out[sel] = pred[sel] * (1 + s * (fac[g] - 1))
    return out


def fit_fac(y, p, Mv):
    return {g: lsf(y[fn(Mv)], p[fn(Mv)]) for g, fn in GROUPS.items()}


print("\n== 2. P -> H (fit on P teacher-level blend, apply to H corrected)")
facP = {m: fit_fac(D[m]["yP"], D[m]["pP"], MP) for m in MET}
for m in MET:
    print(f"  factors P {m}: " + ", ".join(f"{g} {v:.4f}" for g, v in facP[m].items()))
nH = len(H)
B = 2000
aff = np.zeros(nH, bool)
for fn in GROUPS.values():
    aff |= fn(MH)
print(f"  H rows affected: {aff.sum()} (le3 {GROUPS['le3'](MH).sum()}, eq4 {GROUPS['eq4'](MH).sum()})")
# Poisson bootstrap weights over H rows, generated in chunks; keep the full-row SSE per rep and the affected-row weights
SW = np.zeros(B)
SSE0 = {m: np.zeros(B) for m in MET}
WAFF = np.zeros((B, int(aff.sum())))
_rng = np.random.default_rng(12345)
for c0 in range(0, B, 100):
    Wc = _rng.poisson(1.0, size=(100, nH)).astype(np.float64)
    SW[c0:c0 + 100] = Wc.sum(1)
    WAFF[c0:c0 + 100] = Wc[:, aff]
    for m in MET:
        SSE0[m][c0:c0 + 100] = Wc @ ((D[m]["yH"] - D[m]["cH"]) ** 2)
    del Wc


def total_delta(newp, w=None):
    tot = 0.0
    per = {}
    for m in MET:
        d = D[m]
        a = nrmse(d["yH"], newp[m], m, w) - nrmse(d["yH"], d["cH"], m, w)
        per[m] = a
        tot += a / 5
    return tot, per


def boot_total(newp):
    """Poisson-bootstrap distribution of the total delta with fixed new predictions (only affected rows change)"""
    out = np.zeros(B)
    for m in MET:
        d = D[m]
        r0 = (d["yH"] - d["cH"]) ** 2
        r1 = (d["yH"] - newp[m]) ** 2
        sse1 = SSE0[m] + WAFF @ (r1[aff] - r0[aff])
        out += (np.sqrt(sse1 / SW) - np.sqrt(SSE0[m] / SW)) / SIG[m] / 5
    return out


for s in (0.25, 0.5, 0.75, 1.0):
    newp = {m: apply_rule(D[m]["cH"], MH, facP[m], s) for m in MET}
    tot, per = total_delta(newp)
    bt = boot_total(newp)
    print(f"  shrink {s:.2f}: total {tot:+.8f} | per-metric NRMSE " + " ".join(f"{m} {per[m]:+.7f}" for m in MET)
          + f" | boot sd {bt.std():.8f} P(<0) {np.mean(bt < 0):.3f} 5-95% [{np.quantile(bt, .05):+.7f}, {np.quantile(bt, .95):+.7f}]")

print("\n  P -> H with factor estimation noise (bootstrap P group rows for f, bootstrap H rows for evaluation), shrink 0.5")
rng2 = np.random.default_rng(777)
draws = []
idxP = {g: np.flatnonzero(fn(MP)) for g, fn in GROUPS.items()}
for b in range(1000):
    fac_b = {}
    for m in MET:
        fac_b[m] = {}
        for g in GROUPS:
            ii = rng2.choice(idxP[g], size=len(idxP[g]), replace=True)
            fac_b[m][g] = lsf(D[m]["yP"][ii], D[m]["pP"][ii])
    newp = {m: apply_rule(D[m]["cH"], MH, fac_b[m], 0.5) for m in MET}
    w = rng2.poisson(1.0, size=nH).astype(np.float64)
    tot, _ = total_delta(newp, w)
    draws.append(tot)
draws = np.array(draws)
print(f"  total mean {draws.mean():+.8f} sd {draws.std():.8f} P(<0) {np.mean(draws < 0):.3f} "
      f"5-95% [{np.quantile(draws, .05):+.8f}, {np.quantile(draws, .95):+.8f}]")

print("\n== 3. H internal repeated random 2-fold cross-fit (fit f on one half of H corrected, apply to other half)")
for s in (0.5, 1.0):
    res = []
    pm = {m: [] for m in MET}
    for rep in range(300):
        perm = np.random.default_rng(1000 + rep).permutation(nH)
        half = np.zeros(nH, bool)
        half[perm[: nH // 2]] = True
        newp = {}
        for m in MET:
            d = D[m]
            out = d["cH"].copy()
            for fit_mask in (half, ~half):
                fac = fit_fac(d["yH"][fit_mask], d["cH"][fit_mask], MH[fit_mask])
                app = ~fit_mask
                out[app] = apply_rule(d["cH"][app], MH[app], fac, s)
            newp[m] = out
        tot, per = total_delta(newp)
        res.append(tot)
        for m in MET:
            pm[m].append(per[m])
    res = np.array(res)
    print(f"  shrink {s:.2f}: mean total {res.mean():+.8f} sd over splits {res.std():.8f} frac<0 {np.mean(res < 0):.3f} | "
          + " ".join(f"{m} {np.mean(pm[m]):+.7f}" for m in MET))

print("\n== 3b. H internal repeated 5-fold cross-fit, shrink 0.5")
res = []
for rep in range(100):
    fold = np.random.default_rng(5000 + rep).integers(0, 5, nH)
    newp = {}
    for m in MET:
        d = D[m]
        out = d["cH"].copy()
        for k in range(5):
            fm = fold != k
            fac = fit_fac(d["yH"][fm], d["cH"][fm], MH[fm])
            out[~fm] = apply_rule(d["cH"][~fm], MH[~fm], fac, 0.5)
        newp[m] = out
    res.append(total_delta(newp)[0])
res = np.array(res)
print(f"  mean total {res.mean():+.8f} sd {res.std():.8f} frac<0 {np.mean(res < 0):.3f}")

print("\n== 4. H -> P (reverse): fit on H corrected (and on H teacher), apply to the P-side blend, scored on P")
for src in ("cH", "tH"):
    facH = {m: fit_fac(D[m]["yH"], D[m][src], MH) for m in MET}
    for s in (0.5, 1.0):
        tot = 0.0
        per = {}
        for m in MET:
            d = D[m]
            new = apply_rule(d["pP"], MP, facH[m], s)
            per[m] = nrmse(d["yP"], new, m) - nrmse(d["yP"], d["pP"], m)
            tot += per[m] / 5
        print(f"  fit on H {src} shrink {s:.2f}: total on P {tot:+.8f} | " + " ".join(f"{m} {per[m]:+.7f}" for m in MET)
              + " | factors " + "; ".join(f"{m} " + "/".join(f"{v:.3f}" for v in facH[m].values()) for m in MET))

print("\n== 5. public-size subsets of H (55,196 rows, without replacement), P-fit factors, shrink 0.5")
newp = {m: apply_rule(D[m]["cH"], MH, facP[m], 0.5) for m in MET}
pub = []
rng3 = np.random.default_rng(99)
for b in range(2000):
    w = np.zeros(nH, np.float32)
    w[rng3.choice(nH, 55196, replace=False)] = 1
    pub.append(total_delta(newp, w)[0])
pub = np.array(pub)
print(f"  public-size: mean {pub.mean():+.8f} sd {pub.std():.8f} P(<0) {np.mean(pub < 0):.3f} "
      f"P(<= -0.00001) {np.mean(pub <= -1e-5):.3f} 5-95% [{np.quantile(pub, .05):+.8f}, {np.quantile(pub, .95):+.8f}]")

print("\n== 6. leave-one-row-out influence on H (shrink 0.5, P factors): top contributing rows to the delta")
for m in MET:
    d = D[m]
    new = apply_rule(d["cH"], MH, facP[m], 0.5)
    dsse = (d["yH"] - new) ** 2 - (d["yH"] - d["cH"]) ** 2
    idx = np.flatnonzero(aff)
    contrib = dsse[idx]
    order = np.argsort(contrib)
    tot_sse = np.sum((d["yH"] - d["cH"]) ** 2)
    # delta NRMSE ~ dSSE / (2 * SSE) * NRMSE
    base = nrmse(d["yH"], d["cH"], m)
    scale = base / (2 * tot_sse)
    print(f"  {m}: sum dSSE-> dNRMSE {contrib.sum() * scale:+.7f}; best 5 rows {np.round(contrib[order[:5]] * scale, 8).tolist()}; "
          f"worst 5 rows {np.round(contrib[order[-5:]] * scale, 8).tolist()}; share of rows improved {np.mean(contrib < 0):.2f}; "
          f"delta without best 5 {(contrib.sum() - contrib[order[:5]].sum()) * scale:+.7f}")
