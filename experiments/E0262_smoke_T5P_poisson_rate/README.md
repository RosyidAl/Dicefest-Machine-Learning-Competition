# E0262_smoke T5P_poisson_rate

- Hipotesis: Poisson pada laju per meter dengan bobot M (GLM laju dengan exposure) mengalahkan E0260
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27278 (fold )
  - target_macro_elec_kwh: 0.33423 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69380 (fold [0.33639, 0.33209])
- Runtime 2.1 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
