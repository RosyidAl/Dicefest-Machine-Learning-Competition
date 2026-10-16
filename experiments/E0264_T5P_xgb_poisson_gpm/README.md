# E0264 T5P_xgb_poisson_gpm

- Hipotesis: Poisson XGB pada gas per meter (bobot M, offset log M), mcw setara 50 baris; vs E0190 (MSE rasio bobot M^2/A)
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25085 (fold )
  - target_eui_gas_kwh_per_sqm: 0.66054 (fold [0.25107, 0.25214, 0.25147, 0.25059, 0.24895])
- Runtime 35.0 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
