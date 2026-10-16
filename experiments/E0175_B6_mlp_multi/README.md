# E0175 B6_mlp_multi

- Hipotesis: B6: MLP multi-output sederhana (5 target terstandar, 20 epoch one-cycle) [CV primary]
- Setup: model mlp_multi, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2, 3, 4], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.26541 (fold )
  - target_macro_elec_kwh: 0.32759 (fold )
  - target_eui_gas_kwh_per_sqm: 0.69025 (fold )
  - target_clean_building_ratio: 0.05393 (fold )
  - target_electric_heating_adoption: 0.14776 (fold [0.14958, 0.14954, 0.14557, 0.14715, 0.14692])
- Runtime 1.6 menit, peak RAM 0.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
