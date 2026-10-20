# Independent reproduction of small-meter (bld_dim_12<=4) multiplicative fix.
import numpy as np, json
d = np.load('/root/Dicefest/exp/audit/chain_H.npz', allow_pickle=False)
P, H, M = d['P'], d['H'], d['M']
SIG = {'gas': 151750.04, 'eui': 40.65, 'elec': 41725.37}
MH, MP = M[H], M[P]
groups = [lambda m: m <= 3, lambda m: m == 4]
def nrmse(y, p, s): return np.sqrt(np.mean((y - p) ** 2)) / s
def fit_f(y, p): return float(y @ p / (p @ p))
def apply(pred, m, facs, shrink):
    out = pred.copy()
    for g, f in zip(groups, facs):
        k = g(m); out[k] = pred[k] * (1 + shrink * (f - 1))
    return out
res = {}
# (a) P -> H (replicates claim's protocol)
for shrink in [0.5, 0.75, 1.0]:
    tot = 0; per = {}
    for t in SIG:
        yP = d[t+'_pP'] + d[t+'_rP']; pP = d[t+'_pP']
        facs = [fit_f(yP[g(MP)], pP[g(MP)]) for g in groups]
        y, c = d[t+'_y'], d[t+'_corrected']
        new = np.maximum(apply(c, MH, facs, shrink), 0)
        dl = nrmse(y, new, SIG[t]) - nrmse(y, c, SIG[t])
        per[t] = (facs, dl); tot += dl / 5
    res[f'PtoH_s{shrink}'] = {'total': tot, 'per': per}
# (b) cross-fit within H only (independent of P), many seeds, K folds; factors from corrected preds
def crossfit_H(K, seed, shrink, useP=False):
    rng = np.random.default_rng(seed); fold = rng.integers(0, K, len(H))
    tot = 0; per = {}
    for t in SIG:
        y, c = d[t+'_y'], d[t+'_corrected']; new = c.copy()
        for k in range(K):
            tr = fold != k; te = fold == k
            facs = []
            for g in groups:
                gm = g(MH) & tr
                yy, pp = y[gm], c[gm]
                if useP:
                    gp = g(MP); yy = np.r_[yy, d[t+'_pP'][gp] + d[t+'_rP'][gp]]; pp = np.r_[pp, d[t+'_pP'][gp]]
                facs.append(fit_f(yy, pp))
            new[te] = apply(c[te], MH[te], facs, shrink)
        new = np.maximum(new, 0)
        dl = nrmse(y, new, SIG[t]) - nrmse(y, c, SIG[t]); per[t] = dl; tot += dl / 5
    return tot, per
for K in [2, 5]:
    for shrink in [0.5, 1.0]:
        for useP in [False, True]:
            tots = [crossfit_H(K, s, shrink, useP) for s in range(40)]
            arr = np.array([x[0] for x in tots])
            pert = {t: float(np.mean([x[1][t] for x in tots])) for t in SIG}
            res[f'Hcf_K{K}_s{shrink}_P{int(useP)}'] = {'mean_total': arr.mean(), 'sd_over_seeds': arr.std(), 'frac_neg': float((arr < 0).mean()), 'per_mean': pert}
# (c) bootstrap over H rows of P->H s0.5 delta (row-resample uncertainty)
facsP = {t: [fit_f((d[t+'_pP']+d[t+'_rP'])[g(MP)], d[t+'_pP'][g(MP)]) for g in groups] for t in SIG}
rng = np.random.default_rng(123); sq = {}
for t in SIG:
    y, c = d[t+'_y'], d[t+'_corrected']; new = np.maximum(apply(c, MH, facsP[t], 0.5), 0)
    sq[t] = ((y-c)**2, (y-new)**2)
bs = []
for b in range(500):
    i = rng.integers(0, len(H), len(H)); tot = 0
    for t in SIG:
        a, n = sq[t]; tot += (np.sqrt(n[i].mean()) - np.sqrt(a[i].mean())) / SIG[t] / 5
    bs.append(tot)
bs = np.array(bs); res['boot_PtoH_s0.5'] = {'sd': bs.std(), 'frac_neg': float((bs < 0).mean()), 'q05': float(np.quantile(bs, .05)), 'q95': float(np.quantile(bs, .95))}
# bias diagnostics in H corrected
diag = {}
for t in SIG:
    y, c = d[t+'_y'], d[t+'_corrected']
    for name, g in zip(['M<=3', 'M=4', 'M=5'], groups + [lambda m: m == 5]):
        k = g(MH); r = y[k] / np.maximum(c[k], 1e-9)
        diag[f'{t}_{name}'] = {'n': int(k.sum()), 'ratio_sum': float(y[k].sum() / c[k].sum()), 'f_ls': fit_f(y[k], c[k])}
res['diag_H'] = diag
json.dump(res, open('/root/Dicefest/exp/audit/verify2_tried/smallM_v3/repro.json', 'w'), indent=1, default=float)
for k, v in res.items(): print(k, json.dumps(v, default=lambda x: round(float(x), 7) if not isinstance(x, list) else x)[:400])
