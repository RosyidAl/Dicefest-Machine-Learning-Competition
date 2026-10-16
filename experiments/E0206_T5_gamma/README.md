# E0206 T5_gamma

- Hipotesis: T5 objective Gamma [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25957 (fold )
  - target_macro_elec_kwh: 0.32978 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67437 (fold [0.67394, 0.67465, 0.67451])
- Runtime 8.9 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
