# E0200 T0_identity

- Hipotesis: T0 identity (acuan Fase 3) untuk t1 gas, t2 elec, t3 EUI [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_macro_gas_kwh: 0.25877 (fold )
  - target_macro_elec_kwh: 0.32830 (fold )
  - target_eui_gas_kwh_per_sqm: 0.67436 (fold [0.67508, 0.67433, 0.67366])
- Runtime 10.5 menit, peak RAM 5.6 GB
- Kesimpulan: lihat reports/JOURNAL.md
