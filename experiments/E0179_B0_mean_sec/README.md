# E0179 B0_mean_sec

- Hipotesis: B0: prediksi rata-rata fold training (sanity sekitar 1,0) [CV secondary]
- Setup: model mean, fitur raw+, transform {"default": {"type": "identity"}}, CV data/folds/folds_secondary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 1.00034 (fold )
  - target_macro_elec_kwh: 1.00022 (fold )
  - target_eui_gas_kwh_per_sqm: 1.00088 (fold )
  - target_clean_building_ratio: 1.00072 (fold )
  - target_electric_heating_adoption: 1.00082 (fold [0.98861, 0.95728, 1.00127, 1.03927, 1.01585])
- Runtime 0.1 menit, peak RAM 3.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
