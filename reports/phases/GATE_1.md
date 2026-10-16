# GATE 1 (2026-10-16 12:40 WIB): Fase 0 dan 1 selesai, D001 tertulis

## Skor terbaik
| Target | Skor | Sumber |
|---|---|---|
| t1 gas | 0,24591 | v15nc |
| t2 elec | 0,31641 | v15nc |
| t3 EUI | 0,65211 | v15nc |
| t4 clean | 0 | identitas eksak |
| t5 adopsi | sekitar 0,036 | |
| Rata-rata | 0,25002 | honest nested CV, ENS0017, sebelum koreksi residual |

- Koreksi residual kNN (PP0004) di holdout: sekitar -0,0009.
- Public LB terbaik terkonfirmasi: 0,2499 (sub_v15ncrd).
- BEST_CURRENT: sub_v15ncrkd (ekspektasi LB sekitar 0,2494).

## Selesai
- Fase 0 (lingkungan, data, metrik, registry).
- Fase 1 (EDA_REPORT, D001 split acak per baris).
- Pipeline final terbentuk (final/run_all.py). Uji drift identik byte untuk XGB, MLP, dan LGB; inference dari model tersimpan juga identik.

## Berjalan
- Antrean GPU: baseline B4 sampai B6, FB07 A/B, eksperimen coverage Fase 5/7/9.
- Antrean CPU: baseline B0 sampai B3, screening transform Fase 3, ablation Fase 4, SHAP.

## Risiko
- GPU dibagi dengan container lain (VRAM sisa sekitar 5 GB).
- Tiga run audit butuh sekitar 60 jam. Mitigasi: freeze dimajukan (D003) dan pruning anggota menghemat 6+ jam per run tanpa kehilangan skor.
