# E0171_smoke B2_lgb_raw

- Hipotesis: B2: LightGBM referensi (Lampiran H, lr 0.1 screening) tanpa ID spasial, lima target identity
- Setup: model lgb, fitur raw+, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.28971 (fold )
  - target_macro_elec_kwh: 0.34784 (fold )
  - target_eui_gas_kwh_per_sqm: 0.73358 (fold )
  - target_clean_building_ratio: 0.04181 (fold )
  - target_electric_heating_adoption: 0.19436 (fold [0.20054, 0.18865])
- Runtime 2.3 menit, peak RAM 2.9 GB
- Kesimpulan: lihat reports/JOURNAL.md
