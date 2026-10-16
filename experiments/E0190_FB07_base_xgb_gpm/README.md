# E0190 FB07_base_xgb_gpm

- Hipotesis: Acuan FB07: XGB gpm (x4 params) FS_legacy + agg + TE, 5-fold
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25291 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66444 (fold [0.25262, 0.25458, 0.25355, 0.25271, 0.25109])
- Runtime 5.7 menit, peak RAM 5.4 GB
- Kesimpulan: lihat reports/JOURNAL.md
