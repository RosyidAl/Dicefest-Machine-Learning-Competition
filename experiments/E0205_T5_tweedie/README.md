# E0205 T5_tweedie

- Hipotesis: T5 objective Tweedie (variance power 1.5) [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25788 (fold )
  - target_macro_elec_kwh: 0.32791 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67381 (fold [0.67485, 0.6735, 0.67308])
- Runtime 9.1 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
