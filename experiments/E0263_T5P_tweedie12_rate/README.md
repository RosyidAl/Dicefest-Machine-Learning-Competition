# E0263 T5P_tweedie12_rate

- Hipotesis: Tweedie p=1.2 pada laju per meter (bobot M) vs E0260
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25629 (fold )
  - target_macro_elec_kwh: 0.32587 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67030 (fold [0.32618, 0.3259, 0.32552])
- Runtime 15.9 menit, peak RAM 5.5 GB
- Kesimpulan: lihat reports/JOURNAL.md
