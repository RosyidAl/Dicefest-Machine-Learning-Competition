# STATUS (diperbarui: 2026-10-17 09:00 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: Fase 11 (run #1 final v16) + Fase 3/8 (A/B Poisson) / G2 17 Okt malam
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB.
Job berjalan:
- run #1 final v16: unit systemd dice-run1 (resume 08:55 WIB setelah job mati serentak 04:05), log logs/run1.log + final/work/run1/lane_*.log, output final/output/submission_run1.csv
- A/B Poisson E0260-E0265 SELESAI: Poisson menang (LGB E0262, XGB E0264/E0265)
- anggota Poisson v17 (GPU): unit systemd dice-v17, logs/v17_pois_chain.sh, status logs/v17_pois_guard.log, log logs/v17_<mode>_<name>.log
- queue worker CPU/GPU: STOP (sengaja, selama run audit)
Cek job: systemctl status dice-run1 dice-v17. Selalu luncurkan job panjang dengan systemd-run (nohup setsid ikut terbunuh).
Catatan lingkungan: nvidia-smi gagal (NVML 595 vs kernel 580), tetapi torch/XGBoost CUDA berjalan (A4000, 16,6 GB free).
Terbaik CV_PRIMARY: v16 blend honest 0,242467 (34 anggota) | t1 0,2452 | t2 0,3157 | t3 0,6517 | t4 0 | t5 sekitar 0,036
Holdout rantai penuh v16 (blend + koreksi grup/kNN): gas 0,24539, EUI 0,64961, elec 0,31477 (ekspektasi LB sekitar 0,2490)
BEST_CURRENT.csv: sub_v16dev, sha256 e9375ba6...9a7f, public LB 0,24943 (terbaik; v15ncrkd 0,24954)
Kompetitor di public LB: 0,23666 (user, 17 Okt dini hari).
5 langkah berikutnya:
1. Pantau run #1 (final/work/run1/lane_*.log); setelah selesai bandingkan submission_run1.csv dengan sub_v16dev (sha256 harus identik)
2. Anggota Poisson v17 (src/exp_v17_pois.py): rantai GPU logs/v17_pois_chain.sh (xp_gpm, xp_eui, xp_epm; full + hold), status logs/v17_pois_guard.log; lp_gpm, lp_epm setelah jalur CPU run #1 selesai
3. Holdout: fit ulang λ (src/resid_knn.py fit), seleksi anggota (exp/analysis/select_v16.py versi v17), bandingkan rantai penuh v17 vs v16
4. Bila v17 lebih baik: freeze v17 (members.yaml + resid_knn.json), notebook headless + run_all, paling lambat mulai 18 Okt
5. Paket final, laporan Fase 12
Risiko / blocker / NEEDS_USER: TEAM_NAME belum diisi; format writeup di guidebook Drive tidak terbaca
