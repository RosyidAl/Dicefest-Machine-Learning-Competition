# E0243_smoke A01_pseudo_label

- Hipotesis: A01: pseudo-label test (teacher per fold, bobot 0,5) menurunkan NRMSE XGB gpm vs E0190 fold 0-2
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27611 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69822 (fold [0.27984, 0.27239])
- Runtime 1.8 menit, peak RAM 5.8 GB
- Kesimpulan: lihat reports/JOURNAL.md
