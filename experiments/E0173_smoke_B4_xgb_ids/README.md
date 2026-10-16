# E0173_smoke B4_xgb_ids

- Hipotesis: B4: XGBoost GPU parameter wajar + id numerik [CV primary]
- Setup: model xgb, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.28789 (fold )
  - target_macro_elec_kwh: 0.34559 (fold )
  - target_eui_gas_kwh_per_sqm: 0.72707 (fold )
  - target_clean_building_ratio: 0.03461 (fold )
  - target_electric_heating_adoption: 0.20965 (fold [0.21294, 0.20638])
- Runtime 0.6 menit, peak RAM 3.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
