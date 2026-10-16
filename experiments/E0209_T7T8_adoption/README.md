# E0209 T7T8_adoption

- Hipotesis: T7 proporsi + cross_entropy vs T8 logit untuk t5 (lihat E0210); t5 identity sebagai acuan [screening: LGB lr 0.1, 3 fold]
- Setup: model lgb, fitur legacy+agg+te, transform {"default": {"type": "identity"}}, CV data/folds/folds_primary_k5_s42.parquet folds [0, 1, 2], seed [42]
- Hasil (NRMSE OOF):
  - target_electric_heating_adoption: 0.07481 (fold [0.07632, 0.07512, 0.07303])
- Runtime 90.2 menit, peak RAM 6.0 GB
- Kesimpulan: lihat reports/JOURNAL.md
