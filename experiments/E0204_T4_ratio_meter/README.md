# E0204 T4_ratio_meter

- Hipotesis: T4 rasio per meter (bld_dim_12) dengan bobot E^2 untuk t1 dan t2; t3 via EUI identity [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 2}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25775 (fold )
  - target_macro_elec_kwh: 0.32675 (fold [0.32648, 0.32765, 0.32613])
- Runtime 6.4 menit, peak RAM 5.5 GB
- Kesimpulan: lihat reports/JOURNAL.md
