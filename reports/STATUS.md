# STATUS (diperbarui: 2026-10-16 11:28 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: Fase 1 (EDA dan forensik data) / G1 16 Okt malam
Fase selesai: Fase 0 (2026-10-16 11:28)
Job berjalan: tidak ada
Terbaik CV_PRIMARY: mean 0,25001 (ENS0016 v15) / 0,25002 (ENS0017 v15nc, deterministik) | t1 0,24591 | t2 0,31641 | t3 0,65211 | t4 0,00000 | t5 sekitar 0,036 (sumber: exp/prune_v15nc.log); + koreksi residual kNN (PP0004, holdout sekitar -0,0009 total)
BEST_CURRENT.csv: sub_v15ncrkd (v15nc + koreksi kNN + dup_fix), sha256 b6431a33...d8ba8d2, public LB belum (terbaik terkonfirmasi: sub_v15ncrd 0,2499)
5 langkah berikutnya:
1. Fase 1: EDA_REPORT.md + findings.json + figur (forensik split, adversarial validation, target, fitur, struktur fisik)
2. Fase 2: baseline B0-B6 di CV_PRIMARY/SECONDARY; uji determinisme PLR/TabM/multi-task
3. Pruning sadar runtime anggota final; bangun final/run_all.py
4. FB07 KNN-target di dalam model (kandidat gain terbesar)
5. Freeze 18 Okt 18:00 (D003), lalu run #1
Risiko / blocker / NEEDS_USER: GPU dibagi dengan container lain (sekitar 5 GB VRAM sisa); 3 run audit sekitar 60 jam; TEAM_NAME belum diisi
