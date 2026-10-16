# E0173 B4_xgb_ids

- Hipotesis: B4: XGBoost GPU parameter wajar + id numerik [CV primary]
- Setup: model xgb, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.26309 (fold )
  - target_macro_elec_kwh: 0.33009 (fold )
  - target_eui_gas_kwh_per_sqm: 0.68645 (fold )
  - target_clean_building_ratio: 0.01575 (fold )
  - target_electric_heating_adoption: 0.09502 (fold [0.09769, 0.09448, 0.09349, 0.09359, 0.09595])
- Runtime 35.6 menit, peak RAM 4.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
