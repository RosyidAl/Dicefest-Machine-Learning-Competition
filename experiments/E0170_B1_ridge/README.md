# E0170 B1_ridge

- Hipotesis: B1: Ridge (StandardScaler, log1p fitur miring) dengan ID numerik [CV primary]
- Setup: model ridge, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.31137 (fold )
  - target_macro_elec_kwh: 0.35558 (fold )
  - target_eui_gas_kwh_per_sqm: 0.79288 (fold )
  - target_clean_building_ratio: 0.26483 (fold )
  - target_electric_heating_adoption: 0.27587 (fold [0.28293, 0.28177, 0.27865, 0.27641, 0.27908])
- Runtime 2.2 menit, peak RAM 3.1 GB
- Kesimpulan: lihat reports/JOURNAL.md
