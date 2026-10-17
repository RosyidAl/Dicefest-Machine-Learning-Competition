# STATUS (diperbarui: 2026-10-17 14:02 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: maksimalkan model (permintaan user 17 Okt: notebook belakangan) + run #1 final v16 di latar / G2 17 Okt malam
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan:
- run #1 final v16: unit systemd dice-run1, jalur GPU dilanjutkan 11:11 (dijeda 02:24-11:11 untuk v17), jalur CPU selesai. Log logs/run1.log + final/work/run1/lane_*.log, output final/output/submission_run1.csv. Sisa: 14 anggota XGB + 7 NN, ETA sekitar 21:30 WIB.
- SHAP selesai (reports/figures/shap_*). Uji CV sekunder E0271-E0278: unit dice-cv2, log logs/cv_secondary.log, hasil exp/cv2/results.jsonl
- Job lain /root/kaggle (pid 141266 dan 144343, train_noisy_student.py) DIJEDA dengan SIGSTOP atas izin user; lanjutkan setelah run final: kill -CONT 141266 144343
- FB07b (kNN-target diperluas): smoke dice-fb7s (logs/fb07b_smoke.log), lalu A/B penuh src/exp_fb07b.py base|ext; anggota kandidat src/exp_v18_knn2.py
- Disk 28 GB bebas setelah cache pip/uv dihapus (izin user)
- queue worker CPU/GPU: STOP (sengaja, selama run audit)
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
