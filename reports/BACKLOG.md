# BACKLOG

| Prio | Ide | Estimasi gain | Biaya | Status |
|---|---|---|---|---|
| P0 | Pipeline final reproducible (final/run_all.py + notebook) dari anggota v15nc + koreksi | wajib | sekitar 20 jam per run | QUEUED |
| P0 | Uji determinisme PLR / TabM / multi-task (R05) | wajib | 30 menit GPU | QUEUED |
| P0 | Pruning sadar runtime: buang anggota mahal (p1_gpm 3,2 jam, t1_lepm 2,9 jam) bila rugi < 0,00002 | hemat sekitar 6 jam per run | 10 menit CPU | QUEUED |
| P1 | FB07 fitur KNN-target OOF di dalam model (sinyal kNN residual terbukti di PP0004) | -0,0002 sampai -0,0005? | 2-4 jam | QUEUED |
| P1 | Baseline B0-B6 di CV_PRIMARY dan CV_SECONDARY | laporan | 2-3 jam | QUEUED |
| P1 | Adversarial validation (V03) | laporan | 10 menit | QUEUED |
| P1 | Transform T2/T3/T5/T6 di LGB acuan (screening 3 fold) | kecil | 1-2 jam CPU | QUEUED |
| P2 | FB03/FB08/FB09/FB10/FB12 screening | kecil | 2-3 jam CPU | QUEUED |
| P2 | Model zoo tambahan: RealMLP, GPBoost, ExtraTrees, KNN, Ridge, MultiRMSE, multi_output_tree | keragaman | 4-6 jam | QUEUED |
| P2 | Ensemble: top-N avg, hill climbing, level-2 MLP | kecil | 30 menit | QUEUED |
