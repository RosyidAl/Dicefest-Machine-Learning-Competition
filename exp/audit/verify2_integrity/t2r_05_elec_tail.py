"""verify2_integrity t2r_05: is elec (or EUI) also hard-bounded? top-tail counts from chain_H labels (P + H)."""
import json
import numpy as np
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H = z["P"], z["H"]
out = {}
for m in ("elec", "eui", "gas"):
    y = np.empty(len(z["A"]))
    y[P] = z[f"{m}_pP"] + z[f"{m}_rP"]
    y[H] = z[f"{m}_y"]
    q = np.quantile(y, [0.99, 0.999, 0.9999])
    top = np.sort(y)[-8:]
    mx = y.max()
    # counts in the last 5 bins of width (max - q99.9) / 5: a hard cut shows flat counts up to the max, a smooth tail decays
    e = np.linspace(q[1], mx, 6)
    out[m] = dict(q99=float(q[0]), q999=float(q[1]), q9999=float(q[2]), max=float(mx), top8=top.round(1).tolist(),
                  counts_q999_to_max=np.histogram(y, e)[0].tolist())
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/verify2_integrity/t2r_05_elec_tail.json", "w"), indent=1)
