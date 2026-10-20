import numpy as np
z = np.load('exp/audit/chain_H.npz', allow_pickle=False)
for k in sorted(z.files):
    a = z[k]
    print(k, a.dtype, a.shape, (np.round(np.nanmin(a), 4), np.round(np.nanmax(a), 4)) if a.size and a.dtype.kind in 'fiu' else '')
