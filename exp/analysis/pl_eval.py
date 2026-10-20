"""PL0001 evaluation on the holdout chain: teacher (VC blend of the hold members, applied to H) vs pseudo-labelled
student (src/exp_pl.py chain, out-of-fold predictions of the H rows), alone and inside the residual correction.

The student enters the correction as one more regressor: target = y_H - teacher_H, features = group + kNN correction
features (current configs: gas / elec CONFIGS, eui exp/analysis/corr_final.json) + (student_H - teacher_H); all
weights 5-fold cross-fitted on H (exp/analysis/select_corr.cv_rmse), so the score compares directly with the chain
(gas 0.24514, eui 0.64859, elec 0.31462).

usage: python exp/analysis/pl_eval.py <student run name, e.g. lp2_gpm_w1> [<more runs> ...]
       python exp/analysis/pl_eval.py fit <out.json> gas=run[+run] eui=run elec=run
              joint lstsq on all of H of the correction weights and the student weights beta -> resid_knn json with
              "pl": {metric: [{"name": "pl_<run>", "beta": b}]} (test predictions: exp/pl/deploy/test_<run>.npy)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "exp/analysis")
from corr_est import chain  # noqa: E402
from resid_corr import METRICS  # noqa: E402
from resid_knn import CONFIGS  # noqa: E402
from select_corr import CACHE, cv_rmse  # noqa: E402

GAS_KINDS = ("gpm", "eui", "gpgea", "lgpm", "gas")


def main(runs):
    tr, te, d, P, H, A, res = chain()
    cfgs = {m: [tuple(c) for c in CONFIGS[m]] for m in METRICS}
    cfgs["eui"] = [tuple(c) for c in json.load(open("exp/analysis/corr_final.json"))["eui"]["configs"]]
    stud = {}
    for r in runs:
        o = np.load(f"exp/pl/chain/oof/{r}.npy")
        assert len(o) == len(P) + len(H), (r, len(o))
        stud[r] = o[len(P):]
    for metric in METRICS:
        R = res[metric]
        F = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[metric]])
        sig, rH, pH = R["sig"], R["rH"], R["pH"]
        yH = rH + pH
        own = [r for r in runs if (metric == "elec") != any(k in r.split("_")[1] for k in GAS_KINDS)]
        if not own:
            continue
        S = {r: (stud[r] * A[H] if metric == "gas" else stud[r]) for r in own}
        line = [f"{metric}: teacher {np.sqrt(np.mean(rH ** 2)) / sig:.5f}"]
        for r, s in S.items():
            line.append(f"{r} {np.sqrt(np.mean((yH - s) ** 2)) / sig:.5f}")
        print("  ".join(line), flush=True)
        base = cv_rmse(F, rH, sig)
        msg = [f"   corrected: teacher {base:.5f}"]
        for r, s in S.items():
            v = cv_rmse(np.column_stack([F, s - pH]), rH, sig)
            beta = np.linalg.lstsq(np.column_stack([F, s - pH]), rH, rcond=None)[0][-1]
            msg.append(f"+{r} {v:.5f} ({v - base:+.6f}, beta {beta:.3f})")
        if len(S) > 1:
            v = cv_rmse(np.column_stack([F] + [s - pH for s in S.values()]), rH, sig)
            msg.append(f"+all {v:.5f} ({v - base:+.6f})")
        print("  ".join(msg), flush=True)


def fit(out, spec):
    from resid_knn import KSHR
    tr, te, d, P, H, A, res = chain()
    old = json.load(open("exp/resid_knn_vc_AMp_e.json"))
    cfgs = {m: [tuple(c) for c in old["configs"][m]] for m in METRICS}
    lam, pl = {}, {}
    for metric in METRICS:
        R = res[metric]
        F = np.column_stack([R["G"]] + [np.load(f"{CACHE}/{metric}_{c[0]}_{c[1]}_{c[2]}_{c[3]}.npy") for c in cfgs[metric]])
        runs = spec.get(metric, [])
        S = []
        for r in runs:
            o = np.load(f"exp/pl/chain/oof/{r}.npy")[len(P):]
            S.append((o * A[H] if metric == "gas" else o) - R["pH"])
        X = np.column_stack([F] + S)
        coef = np.linalg.lstsq(X, R["rH"], rcond=None)[0]
        if not runs:
            assert np.allclose(coef, old["lam"][metric], rtol=1e-6, atol=1e-9), metric
        lam[metric] = coef[:F.shape[1]].tolist()
        if runs:
            pl[metric] = [{"name": f"pl_{r}", "beta": float(b)} for r, b in zip(runs, coef[F.shape[1]:])]
        print(f"{metric}: cv {cv_rmse(X, R['rH'], R['sig']):.5f} (no student {cv_rmse(F, R['rH'], R['sig']):.5f}) "
              f"beta {[round(float(b), 4) for b in coef[F.shape[1]:]]}", flush=True)
    json.dump({"lam": lam, "configs": {m: [list(c) for c in v] for m, v in cfgs.items()}, "kshr": KSHR,
               "vc": old["vc"], "pl": pl}, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], {kv.split("=")[0]: kv.split("=")[1].split("+") for kv in sys.argv[3:]})
    else:
        main(sys.argv[1:])
