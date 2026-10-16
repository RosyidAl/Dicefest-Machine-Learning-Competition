# STATUS (diperbarui: 2026-10-16 14:36 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: Fase 2-9 paralel (baseline, screening, coverage) + persiapan freeze (Fase 10/11) / G2 17 Okt malam
Fase selesai: Fase 0, Fase 1
Job berjalan: lihat queue/*/running dan logs/worker_{gpu,cpu}.log
- GPU: anggota FB07 (xk_lepa, pk_gpm, xk_lgpm, xk_gpgea, pk_lepm; full + holdout), lalu eksperimen coverage
- CPU: lk_gpm, lk_lepa (full + holdout), lalu analisis ensemble, baseline, screening T, ablation FB, SHAP, Optuna
Terbaik CV_PRIMARY: blend honest 0,242712 (v15nc + xk_gpm, gas+EUI+elec /5) | t1 0,24560 | t2 0,31641 | t3 0,65155 | t4 0 | t5 sekitar 0,036
Holdout rantai penuh (+ koreksi λ v16a): gas 0,24569, EUI 0,65010, elec 0,31569
BEST_CURRENT.csv: sub_v15ncrkd, sha256 b6431a33...8ba8d2, public LB 0,24954
5 langkah berikutnya:
1. Selesaikan anggota FB07 + holdout; fit ulang λ (resid_knn fit) dengan semua anggota holdout baru
2. Pilih daftar anggota final (v15nc pruned + FB07), perbarui final/configs/members.yaml + resid_knn.json, uji assemble
3. Freeze (target 16 Okt malam / 17 Okt pagi), mulai run #1 (3 jalur paralel)
4. Selama run: dokumentasi Fase 2-9, eksperimen coverage CPU
5. Run #2 + notebook headless, paket final, laporan Fase 12
Risiko / blocker / NEEDS_USER: TEAM_NAME belum diisi; format writeup di guidebook Drive tidak terbaca
