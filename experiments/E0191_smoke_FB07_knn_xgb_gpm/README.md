# E0191_smoke FB07_knn_xgb_gpm

- Hipotesis: FB07: + fitur KNN-target OOF (ruang stok bangunan, window sub_n 30/100/300) menurunkan NRMSE gas dan EUI
- Setup: model xgb, fitur legacy+agg+te+knn_target, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27443 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69331 (fold [0.27815, 0.27072])
- Runtime 1.0 menit, peak RAM 4.8 GB
- Kesimpulan: lihat reports/JOURNAL.md
