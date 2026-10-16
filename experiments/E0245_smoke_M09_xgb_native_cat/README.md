# E0245_smoke M09_xgb_native_cat

- Hipotesis: M09: zone_n sebagai kategori native XGBoost vs numerik (E0190 fold 0-2)
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27979 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70877 (fold [0.28428, 0.2753])
- Runtime 1.2 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
