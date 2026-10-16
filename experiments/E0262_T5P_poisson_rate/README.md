# E0262 T5P_poisson_rate

- Hipotesis: Poisson pada laju per meter dengan bobot M (GLM laju dengan exposure) mengalahkan E0260
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25310 (fold )
  - target_macro_elec_kwh: 0.32385 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66536 (fold [0.3238, 0.32437, 0.32339])
- Runtime 29.9 menit, peak RAM 5.7 GB
- Kesimpulan: lihat reports/JOURNAL.md
