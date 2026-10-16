# E0174_smoke B5_cat_ids

- Hipotesis: B5: CatBoost GPU, zone_n dan sub_n sebagai cat_features [CV primary]
- Setup: model cat, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.28584 (fold )
  - target_macro_elec_kwh: 0.34112 (fold )
  - target_eui_gas_kwh_per_sqm: 0.71760 (fold )
  - target_clean_building_ratio: 0.02977 (fold )
  - target_electric_heating_adoption: 0.16178 (fold [0.16566, 0.15787])
- Runtime 1.1 menit, peak RAM 3.1 GB
- Kesimpulan: lihat reports/JOURNAL.md
