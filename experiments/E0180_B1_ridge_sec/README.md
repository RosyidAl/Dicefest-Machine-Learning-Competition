# E0180 B1_ridge_sec

- Hipotesis: B1: Ridge (StandardScaler, log1p fitur miring) dengan ID numerik [CV secondary]
- Setup: model ridge, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_secondary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.31272 (fold )
  - target_macro_elec_kwh: 0.35649 (fold )
  - target_eui_gas_kwh_per_sqm: 0.79515 (fold )
  - target_clean_building_ratio: 0.26558 (fold )
  - target_electric_heating_adoption: 0.27684 (fold [0.28519, 0.2772, 0.26815, 0.27863, 0.29404])
- Runtime 2.2 menit, peak RAM 3.1 GB
- Kesimpulan: lihat reports/JOURNAL.md
