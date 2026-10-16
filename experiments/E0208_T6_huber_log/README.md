# E0208 T6_huber_log

- Hipotesis: T6 Huber (alpha 0.2) di skala log1p [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "log1p"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25791 (fold )
  - target_macro_elec_kwh: 0.32850 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67357 (fold [0.67448, 0.67404, 0.6722])
- Runtime 12.0 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
