# E0264_smoke T5P_xgb_poisson_gpm

- Hipotesis: Poisson XGB pada gas per meter (bobot M, offset log M), mcw setara 50 baris; vs E0190 (MSE rasio bobot M^2/A)
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27430 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69799 (fold [0.27787, 0.27073])
- Runtime 1.2 menit, peak RAM 5.5 GB
- Kesimpulan: lihat reports/JOURNAL.md
