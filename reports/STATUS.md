# STATUS (diperbarui: 2026-10-17 20:55 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: v18 (arahan user: hasil terbaik) + run #1 final v16 di latar / seleksi v18 18 Okt
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan (semua unit systemd, log status logs/v18_guard.log):
- dice-run1: run #1 final v16 (resume), 29/35, ETA sekitar 00:00
- dice-v18g: XGB full xk2_lgpm -> xk2_lepa -> xp_epa -> xp8_gpm (xk2_gpm, xk2_gpgea selesai), ETA 23:55
- dice-v18c: LGB full lp_epm -> lk2_gpm -> lk2_lepa (lp_gpm selesai: EUI 0,65697 terbaik), ETA 23:40
- dice-v18n: NN full mk2 -> pk2_gpgea -> pk2_lepm, mulai setelah jalur NN run #1 (pid 209657), ETA 05:30
- dice-v18h: holdout XGB (7 anggota), mulai setelah dice-v18c + dice-run1
- dice-v18ch: holdout LGB (4 anggota), mulai setelah dice-v18g + dice-run1 + dice-v18c
- dice-v18nh: holdout NN (3 anggota), mulai setelah dice-v18n
- Besok pagi: seleksi (exp/analysis/select_v18.py), resid_knn fit (exp/resid_knn_v18.json), scripts/assemble_dev.py -> sub_v18dev
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
