# E0241_smoke M18_xgb_multi_output

- Hipotesis: M18: XGB multi_output_tree untuk log gas/meter dan log elec/meter sekaligus vs per target (E0242)
- Setup: model xgb_multi, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "logratio"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27570 (fold )
  - target_macro_elec_kwh: 0.33492 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69570 (fold [0.33624, 0.33362])
- Runtime 1.2 menit, peak RAM 0.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
