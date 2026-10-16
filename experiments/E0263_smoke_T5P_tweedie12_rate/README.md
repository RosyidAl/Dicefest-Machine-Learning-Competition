# E0263_smoke T5P_tweedie12_rate

- Hipotesis: Tweedie p=1.2 pada laju per meter (bobot M) vs E0260
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.27764 (fold )
  - target_macro_elec_kwh: 0.33811 (fold )
  - target_eui_gas_kwh_per_sqm: 0.70192 (fold [0.34006, 0.33618])
- Runtime 1.6 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
