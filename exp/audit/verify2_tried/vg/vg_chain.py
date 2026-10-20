# Independent reproduction of fix G (linear soft cap above knot for gas truncation at 1e6).
import numpy as np, json
from scipy.stats import norm
z = np.load('exp/audit/chain_H.npz', allow_pickle=False)
SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
CAP = 1e6
H, P, A = z['H'], z['P'], z['A']
AH, AP = A[H], A[P]
def nr(p, y, m): return np.sqrt(np.mean((p - y) ** 2)) / SIG[m]
out = {}
gy, gp = z['gas_y'], z['gas_corrected']
ey, ep = z['eui_y'], z['eui_corrected']
print('H gas max y', gy.max(), 'n y>=980k', (gy >= 9.8e5).sum(), 'n pred>1e6', (gp > CAP).sum(), 'n pred>9e5', (gp > 9e5).sum())
print('P side: rP==y-pP check: mean rP', z['gas_rP'].mean(), ' pP max', z['gas_pP'].max())
yP = z['gas_pP'] + z['gas_rP']
print('P y max', yP.max())
# implied gas for eui on H
print('eui identity check on H: corr(ey*AH, ?) n/a; max ey*AH', (ey * AH).max())
base_g, base_e = nr(gp, gy, 'gas'), nr(ep, ey, 'eui')
print('base gas %.6f eui %.6f' % (base_g, base_e))

def fitbeta(p, y, knot):
    h = np.maximum(p - knot, 0); return float(h @ (y - p) / (h @ h))
def applyG(p, beta, knot):
    q = p.copy(); m = p > knot
    q[m] = np.minimum(p[m] + beta * (p[m] - knot), CAP); return q
def score(beta, knot, idx=None):
    g2 = applyG(gp, beta, knot); e2 = applyG(ep * AH, beta, knot) / AH
    if idx is None: idx = np.arange(len(gy))
    dg = nr(g2[idx], gy[idx], 'gas') - nr(gp[idx], gy[idx], 'gas')
    de = nr(e2[idx], ey[idx], 'eui') - nr(ep[idx], ey[idx], 'eui')
    return dg, de, (dg + de) / 5

# 1) beta from P (P-side fitted pred; note pP is in-sample-ish fit on P)
for knot in (8.5e5, 9e5, 9.5e5):
    b = fitbeta(z['gas_pP'], yP, knot)
    dg, de, dt = score(b, knot)
    print('P-fit knot %.0f beta %.4f  dgas %+.7f deui %+.7f total %+.8f' % (knot, b, dg, de, dt))
    out[f'Pfit_{int(knot)}'] = dict(beta=b, dgas=dg, deui=de, total=dt)
# hard clip only
g2 = np.minimum(gp, CAP); e2 = np.minimum(ep * AH, CAP) / AH
hc = ((nr(g2, gy, 'gas') - base_g) + (nr(e2, ey, 'eui') - base_e)) / 5
print('hard clip only total %+.8f' % hc); out['hardclip'] = hc

# 2) cross-fit on H: beta from one half (gas pred + eui-implied pairs pooled? use gas only), scored on other half; many seeds
knot = 9e5
res = []
for seed in range(100, 140):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(gy)); K = 5
    folds = np.array_split(perm, K)
    g2 = gp.copy(); e2 = ep.copy()
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        b = fitbeta(gp[tr], gy[tr], knot)
        g2[te] = applyG(gp[te], b, knot); e2[te] = applyG(ep[te] * AH[te], b, knot) / AH[te]
    dg = nr(g2, gy, 'gas') - base_g; de = nr(e2, ey, 'eui') - base_e
    res.append((dg, de, (dg + de) / 5))
res = np.array(res)
print('H 5-fold crossfit (40 seeds): dgas mean %+.7f deui %+.7f total mean %+.8f sd %.8f min %+.8f max %+.8f' % (*res.mean(0), res[:, 2].std(), res[:, 2].min(), res[:, 2].max()))
out['H5cf'] = dict(mean=res.mean(0).tolist(), sd=float(res[:, 2].std()), frac_neg=float((res[:, 2] < 0).mean()))
# split-half: fit on half A, score on half B (both ways)
sh = []
for seed in range(200, 240):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(gy)); a, bb = perm[: len(perm) // 2], perm[len(perm) // 2:]
    for fi, si in ((a, bb), (bb, a)):
        b = fitbeta(gp[fi], gy[fi], knot); sh.append(score(b, knot, si)[2])
sh = np.array(sh)
print('split-half total mean %+.8f sd %.8f frac<0 %.3f' % (sh.mean(), sh.std(), (sh < 0).mean()))
out['splithalf'] = dict(mean=float(sh.mean()), sd=float(sh.std()), frac_neg=float((sh < 0).mean()))

# 3) bias by bin on H corrected
for lo, hi in ((8e5, 9e5), (9e5, 9.5e5), (9.5e5, 2e6)):
    m = (gp >= lo) & (gp < hi); r = gy[m] - gp[m]
    print('bin [%.0f,%.0f) n %d mean resid %.0f se %.0f' % (lo, hi, m.sum(), r.mean(), r.std() / np.sqrt(max(m.sum(), 1))))
bP = fitbeta(z['gas_pP'], yP, 9e5); g2 = applyG(gp, bP, 9e5)
m = gp >= 9.5e5; print('after fix top bin resid %.0f' % (gy[m] - g2[m]).mean())

# 4) incremental over a truncated-normal soft cap (S3-like): q = E[Y | Y<CAP], Y~N(p, (c p)^2), c fitted on P
def tn(p, c):
    s = np.maximum(c * p, 1.0); a = (CAP - p) / s
    return p - s * norm.pdf(a) / np.maximum(norm.cdf(a), 1e-12)
pP = z['gas_pP']
cs = np.linspace(0.02, 0.3, 57)
lossP = [np.mean((np.minimum(tn(pP, c), CAP) - yP) ** 2) for c in cs]
c = cs[int(np.argmin(lossP))]
g3 = np.minimum(tn(gp, c), CAP); e3 = np.minimum(tn(ep * AH, c), CAP) / AH
d3 = ((nr(g3, gy, 'gas') - base_g) + (nr(e3, ey, 'eui') - base_e)) / 5
print('truncnorm c=%.3f total %+.8f' % (c, d3))
# G applied on top of tn
r3 = g3[g3 > 9e5]
h = np.maximum(np.minimum(tn(pP, c), CAP) - 9e5, 0); b4 = float(h @ (yP - np.minimum(tn(pP, c), CAP)) / (h @ h))
g4 = applyG(g3, b4, 9e5); e4 = applyG(e3 * AH, b4, 9e5) / AH
d4 = ((nr(g4, gy, 'gas') - nr(g3, gy, 'gas')) + (nr(e4, ey, 'eui') - nr(e3, ey, 'eui'))) / 5
print('G on top of truncnorm: beta %.4f incremental total %+.8f' % (b4, d4))
out['tn'] = dict(c=float(c), total=d3, G_incremental=d4, beta_after=b4)
json.dump(out, open('exp/audit/verify2_tried/vg/vg_chain.json', 'w'), indent=1)
