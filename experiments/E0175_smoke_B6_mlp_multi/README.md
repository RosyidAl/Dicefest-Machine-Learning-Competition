# E0175_smoke B6_mlp_multi

- Hipotesis: B6: MLP multi-output sederhana (5 target terstandar, 20 epoch one-cycle) [CV primary]
- Setup: model mlp_multi, fitur raw_ids+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.34664 (fold )
  - target_macro_elec_kwh: 0.40017 (fold )
  - target_eui_gas_kwh_per_sqm: 0.82953 (fold )
  - target_clean_building_ratio: 0.34498 (fold )
  - target_electric_heating_adoption: 0.48178 (fold [0.52112, 0.44055])
- Runtime 0.1 menit, peak RAM 0.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
