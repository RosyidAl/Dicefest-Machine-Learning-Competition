# STATUS (diperbarui: 2026-10-17 18:15 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: v18 (arahan user: hasil terbaik) + run #1 final v16 di latar / seleksi v18 18 Okt
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan:
- run #1 final v16: unit dice-run1 (resume 18:11 setelah OOM GPU 17:40), 25/35 output selesai, ETA sekitar 01:00. Log final/work/run1/lane_*.log
- v18 GPU: dice-v18g (logs/dev_chain.sh, status logs/v18_guard.log): xk2_gpm, xk2_gpgea, xk2_lgpm, xk2_lepa, xp_epa, xp8_gpm (full)
- v18 CPU: dice-v18c (mulai bila cache/knn_x lengkap dan MemAvailable > 10 GB): lp_gpm, lp_epm, lk2_gpm, lk2_lepa
- v18 NN: dice-v18n (mulai setelah jalur NN run #1, pid 209657, selesai): mk2, pk2_gpgea, pk2_lepm
- Job /root/kaggle dihentikan atas izin user 17:40
Cek job: systemctl status dice-run1. Selalu luncurkan job panjang dengan systemd-run (nohup setsid ikut terbunuh).
Terbaik CV_PRIMARY: ENS0018 v16 blend honest 0,24960 (5 target) | t1 0,24524 | t2 0,31608 | t3 0,65101 | t4 0 | t5 0,0357
Holdout rantai penuh v16 (blend + koreksi grup/kNN): gas 0,24539, EUI 0,64961, elec 0,31477
BEST_CURRENT.csv: sub_v16dev, sha256 e9375ba6...9a7f, public LB 0,24943 (peringkat 6 public; klaster legit 0,248-0,250)
Figur Fase 12: reports/figures/fig1-fig5 (src/make_figures.py); outline presentasi: reports/PRESENTATION_OUTLINE.md
5 langkah berikutnya:
1. Pantau run #1 (final/work/run1/lane_*.log); setelah selesai bandingkan submission_run1.csv dengan sub_v16dev (sha256 harus identik)
2. scripts/make_artifacts.sh final/work/run1 (model -> final/artifacts + SHA256SUMS); nama tim -> make_notebooks.py; scripts/run_notebook.sh untuk notebook penuh (final/nbrun_full) dan inference (final/nbrun_inf), keduanya harus memberi sha256 yang sama
3. Writeup: isi bagian 5-8 dan 11 (tabel model, hasil ablation, SHAP, kesimpulan), FINAL_REPORT.md
4. Final B: varian dari run yang sama (mis. tanpa koreksi kNN) atau v15nc; putuskan D-FINAL
5. Paket final + checklist NEEDS_USER paling lambat 23 Okt 12:00
Risiko / blocker / NEEDS_USER: TEAM_NAME = BISMILLAH (konfirmasi user); format writeup di guidebook Drive tidak terbaca
