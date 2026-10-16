# E0203 T3_yeojohnson

- Hipotesis: T3 Yeo-Johnson (lambda di-fit di train fold) menurunkan NRMSE [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "yeojohnson"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25934 (fold )
  - target_macro_elec_kwh: 0.32976 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67527 (fold [0.67609, 0.67459, 0.67513])
- Runtime 9.2 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
