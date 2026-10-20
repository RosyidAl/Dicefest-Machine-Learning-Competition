import pandas as pd, numpy as np
for f in ['sub_v19vce', 'sub_v19nca', 'sub_v19ncapp', 'sub_v19nca2pp']:
    g = pd.read_csv(f'submissions/{f}.csv', usecols=['target_macro_gas_kwh'])['target_macro_gas_kwh'].values
    print(f, 'max %.0f n>1e6 %d n>9e5 %d n>9.5e5 %d' % (g.max(), (g > 1e6).sum(), (g > 9e5).sum(), (g > 9.5e5).sum()))
a = pd.read_csv('submissions/sub_v19nca.csv', usecols=['target_macro_gas_kwh'])['target_macro_gas_kwh'].values
b = pd.read_csv('submissions/sub_v19ncapp.csv', usecols=['target_macro_gas_kwh'])['target_macro_gas_kwh'].values
m = a > 9.5e5
print('nca -> ncapp mean shift on rows a>9.5e5: %.0f (n %d)' % ((b[m] - a[m]).mean(), m.sum()))
