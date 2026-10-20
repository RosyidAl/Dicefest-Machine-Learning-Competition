# Independent reproduction: hard cap gas<=B, EUI<=B/A. Bound B derived from P rows only (honest),
# evaluated on H; also bootstrap and halves; also actual submission file check.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'src')
S = {'gas': 151750.04, 'eui': 40.65, 'elec': 41725.37}
z = np.load('exp/audit/chain_H.npz', allow_pickle=False)
P, H, A = z['P'], z['H'], z['A']
trg = pd.read_csv('data/raw/train.csv', usecols=['target_macro_gas_kwh'])['target_macro_gas_kwh'].values
out = {}
# bound from P only: reconstruct gas on P via m_pP + m_rP (pred + residual = y)
gP = trg[P]
out['pP_plus_rP_vs_y_maxabs'] = float(np.max(np.abs(z['gas_pP'] + z['gas_rP'] - gP)))
out['H_y_matches_train'] = float(np.max(np.abs(trg[H] - z['gas_y'])))
B = float(np.max(gP)); out['B_from_P'] = B; out['n_P'] = int(len(P))
out['P_rows_ge_1e6'] = int((gP >= 1e6).sum())
yg, ye = z['gas_y'], z['eui_y']
out['H_y_gas_max'] = float(yg.max()); out['H_eui_y_times_A_max'] = float((ye * A[H]).max())
def nr(y, p, s): return np.sqrt(np.mean((y - p) ** 2)) / s
res = {}
for m, yy in [('gas', yg), ('eui', ye)]:
    p = z[m + '_corrected']
    cap = B if m == 'gas' else B / A[H]
    pc = np.minimum(p, cap)
    aff = pc < p
    base, new = nr(yy, p, S[m]), nr(yy, pc, S[m])
    # per-row squared-error change (must be <=0 per row given y<=cap)
    d = (yy - pc) ** 2 - (yy - p) ** 2
    rng = np.random.default_rng(12345)
    halves = []
    idx = rng.permutation(len(H)); h1, h2 = idx[: len(H)//2], idx[len(H)//2:]
    for h in (h1, h2):
        halves.append((nr(yy[h], pc[h], S[m]) - nr(yy[h], p[h], S[m])) / 5)
    res[m] = dict(n_aff=int(aff.sum()), base=base, capped=new, delta_total=(new - base) / 5,
                  n_rows_worse=int((d > 1e-9).sum()), y_above_cap=int((yy > cap + 1e-9).sum()),
                  half_deltas_total=halves)
out['H'] = res
# actual submission
sub = pd.read_csv('submissions/sub_v19vce.csv')
te = pd.read_csv('data/raw/test.csv', usecols=['id', 'bld_dim_06'])
assert (te['id'].values == sub['id'].values).all()
At = te['bld_dim_06'].values.astype(float)
g = sub['target_macro_gas_kwh'].values; e = sub['target_eui_gas_kwh_per_sqm'].values
out['test_gas_gt_1e6'] = int((g > 1e6).sum()); out['test_gas_vals'] = sorted(np.round(g[g > 1e6]).tolist())
out['test_gas_gt_B'] = int((g > B).sum())
ex = e - 1e6 / At
out['test_eui_gt_1e6_over_A'] = int((ex > 0).sum()); out['test_eui_excess'] = sorted(np.round(ex[ex > 0], 2).tolist())
out['test_A_min'] = float(np.nanmin(At)); out['test_A_nan'] = int(np.isnan(At).sum())
json.dump(out, open('exp/audit/verify2_integrity/r3_hardcap_verify.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
