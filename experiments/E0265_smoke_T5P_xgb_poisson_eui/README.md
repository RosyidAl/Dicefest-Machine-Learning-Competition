# E0265_smoke T5P_xgb_poisson_eui

- Hipotesis: Poisson XGB pada EUI (laju per luas, bobot A = Poisson gas dengan offset log A), mcw setara 50 baris; vs E0190
- Setup: model xgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_06", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27394 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70118 (fold [0.27751, 0.27039])
- Runtime 1.1 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
