# E0260 T5P_ref_ratio

- Hipotesis: Acuan uji Poisson: LGB gpm (bobot M^2/A) dan epm (bobot M), MSE
- Setup: model lgb, fitur legacy+agg+te, transform {"target_macro_elec_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_12", "weight_col_power": -1, "weight_power": 2}, "target_macro_gas_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25672 (fold )
  - target_macro_elec_kwh: 0.32633 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67119 (fold [0.32626, 0.32626, 0.32648])
- Runtime 17.2 menit, peak RAM 5.5 GB
- Kesimpulan: lihat reports/JOURNAL.md
