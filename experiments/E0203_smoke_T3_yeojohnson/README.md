# E0203_smoke T3_yeojohnson

- Hipotesis: T3 Yeo-Johnson (lambda di-fit di train fold) menurunkan NRMSE [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "yeojohnson"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.28299 (fold )
  - target_macro_elec_kwh: 0.34233 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70495 (fold [0.70381, 0.70606])
- Runtime 7.5 menit, peak RAM 5.7 GB
- Kesimpulan: lihat reports/JOURNAL.md
