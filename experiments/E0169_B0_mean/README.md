# E0169 B0_mean

- Hipotesis: B0: prediksi rata-rata fold training (sanity sekitar 1,0) [CV primary]
- Setup: model mean, fitur raw+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 1.00000 (fold )
  - target_macro_elec_kwh: 1.00000 (fold )
  - target_eui_gas_kwh_per_sqm: 1.00006 (fold )
  - target_clean_building_ratio: 1.00008 (fold )
  - target_electric_heating_adoption: 1.00067 (fold [1.01123, 1.00207, 0.98952, 0.99925, 1.00119])
- Runtime 0.1 menit, peak RAM 3.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
