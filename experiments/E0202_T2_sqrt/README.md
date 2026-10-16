# E0202 T2_sqrt

- Hipotesis: T2 sqrt menurunkan NRMSE dibanding identity [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "sqrt"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25814 (fold )
  - target_macro_elec_kwh: 0.32776 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67432 (fold [0.67478, 0.67451, 0.67367])
- Runtime 9.5 menit, peak RAM 5.7 GB
- Kesimpulan: lihat reports/JOURNAL.md
