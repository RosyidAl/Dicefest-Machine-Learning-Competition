# E0191 FB07_knn_xgb_gpm

- Hipotesis: FB07: + fitur KNN-target OOF (ruang stok bangunan, window sub_n 30/100/300) menurunkan NRMSE gas dan EUI
- Setup: model xgb, fitur legacy+agg+te+knn_target, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25181 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66229 (fold [0.25177, 0.25346, 0.25253, 0.25137, 0.24991])
- Runtime 21.3 menit, peak RAM 5.7 GB
- Kesimpulan: lihat reports/JOURNAL.md
