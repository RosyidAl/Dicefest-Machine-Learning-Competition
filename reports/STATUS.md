# STATUS (diperbarui: 2026-10-17 00:45 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: Fase 11 (run #1 final v16) + Fase 3/8 (A/B Poisson) / G2 17 Okt malam
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB.
Job berjalan:
- run #1 final v16: PID 4476 (lane xgb 4520, nn 4521, cpu 4522), mulai 00:36 WIB 17 Okt, log logs/run1.log + final/work/run1/lane_*.log, output final/output/submission_run1.csv
- A/B Poisson E0260-E0263 (smoke lalu full, LGB 4 thread): logs/poisson_ab.sh, status di logs/poisson_ab_guard.log; berhenti sendiri bila MemAvailable < 3 GB
- queue worker CPU/GPU: STOP (sengaja, selama run audit)
Catatan lingkungan: nvidia-smi gagal (NVML 595 vs kernel 580), tetapi torch/XGBoost CUDA berjalan (A4000, 16,6 GB free).
Terbaik CV_PRIMARY: v16 blend honest 0,242467 (34 anggota) | t1 0,2452 | t2 0,3157 | t3 0,6517 | t4 0 | t5 sekitar 0,036
Holdout rantai penuh v16 (blend + koreksi grup/kNN): gas 0,24539, EUI 0,64961, elec 0,31477 (ekspektasi LB sekitar 0,2490)
BEST_CURRENT.csv: sub_v16dev, sha256 e9375ba6...9a7f, public LB 0,24943 (terbaik; v15ncrkd 0,24954)
Kompetitor di public LB: 0,23666 (user, 17 Okt dini hari).
5 langkah berikutnya:
1. Pantau run #1 (tail final/work/run1/lane_*.log); setelah selesai bandingkan submission_run1.csv dengan sub_v16dev (sha256 harus identik)
2. Baca hasil A/B Poisson (E0260-E0263); bila Poisson menang jelas, buat anggota Poisson (full + holdout) untuk kandidat v17
3. Analisis konsentrasi error (scratchpad err_conc.py) untuk mencari sumber gain besar (gap ke 0,2367)
4. Putuskan v17 vs v16 paling lambat 19 Okt pagi (D003: 3 run x ~16-20 jam)
5. Run #2 + notebook headless, paket final, laporan Fase 12
Risiko / blocker / NEEDS_USER: TEAM_NAME belum diisi; format writeup di guidebook Drive tidak terbaca
