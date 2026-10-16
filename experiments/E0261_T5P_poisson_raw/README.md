# E0261 T5P_poisson_raw

- Hipotesis: Poisson pada gas dan elec mentah (tanpa bobot) mengalahkan formulasi rasio MSE (E0260)
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25468 (fold )
  - target_macro_elec_kwh: 0.32543 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66755 (fold [0.32552, 0.32556, 0.32523])
- Runtime 31.7 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
