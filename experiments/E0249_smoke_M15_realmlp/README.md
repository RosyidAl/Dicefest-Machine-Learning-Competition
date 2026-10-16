# E0249_smoke M15_realmlp

- Hipotesis: M15: RealMLP-TD (pytabkit) log gas/meter dan log elec/meter vs XGB (E0242)
- Setup: model realmlp, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "logratio"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27846 (fold )
  - target_macro_elec_kwh: 0.33614 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70235 (fold [0.33812, 0.33418])
- Runtime 18.8 menit, peak RAM 6.2 GB
- Kesimpulan: lihat reports/JOURNAL.md
