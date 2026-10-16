# E0260_smoke T5P_ref_ratio

- Hipotesis: Acuan uji Poisson: LGB gpm (bobot M^2/A) dan epm (bobot M), MSE
- Setup: model lgb, fitur legacy+agg+te, transform {"target_macro_elec_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_12", "weight_col_power": -1, "weight_power": 2}, "target_macro_gas_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27900 (fold )
  - target_macro_elec_kwh: 0.33859 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70541 (fold [0.33989, 0.33731])
- Runtime 1.4 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
