# E0207 T5_poisson

- Hipotesis: T5 objective Poisson [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25468 (fold )
  - target_macro_elec_kwh: 0.32543 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66947 (fold [0.6694, 0.66971, 0.66929])
- Runtime 17.7 menit, peak RAM 5.7 GB
- Kesimpulan: lihat reports/JOURNAL.md
