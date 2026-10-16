# E0201 T1_log1p

- Hipotesis: T1 log1p (expm1 tanpa koreksi bias) menurunkan NRMSE dibanding identity [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "log1p"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.26049 (fold )
  - target_macro_elec_kwh: 0.33056 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67734 (fold [0.6777, 0.67683, 0.67749])
- Runtime 9.3 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
