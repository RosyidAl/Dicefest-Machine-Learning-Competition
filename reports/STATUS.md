# STATUS (diperbarui: 2026-10-18 11:45 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: v18 (arahan user: hasil terbaik) + run #1 final v16 di latar / seleksi v18 18 Okt
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan: TIDAK ADA. Server dipakai user untuk proyek kaggle sejak 11:45 18 Okt (permintaan user).
- Saat dilanjutkan: run final v19 (resume final/work/run1; baru: lk2_gpm, lp_gpm, lp_epm, xp82_gpm, xp2_gpgea, xp2_eui, xp2_epa, lp2_gpm, lp2_epm, pk2_lgpm) -> uji drift vs exp/oof|pred -> sha256 vs sub_v19dev (62444504...); lalu make_notebooks.py BISMILLAH + scripts/run_notebook.sh (full + inference)
- Batas: server harus kembali paling lambat Selasa 20 Okt malam (run final sekitar 8 jam + notebook sekitar 30 jam)
Cek job: systemctl status dice-run1. Selalu luncurkan job panjang dengan systemd-run (nohup setsid ikut terbunuh).
Terbaik CV_PRIMARY: v19 blend nested 0,24937 (5 target; v18 0,24940, v16 0,24960) | t1 0,24498 | t2 0,31589 | t3 0,65029 | t4 0 | t5 0,0357
Holdout rantai penuh v19: gas 0,24531, EUI 0,64930, elec 0,31475 (v18 0,24533 / 0,64937 / 0,31476)
BEST_CURRENT.csv: sub_v19dev (62444504...f216), belum di-submit; LB terbaik sub_v18dev 0,24936. sub_v16fin = output pipeline final v16 (3767ba10...)
Figur Fase 12: reports/figures/fig1-fig5 (src/make_figures.py); outline presentasi: reports/PRESENTATION_OUTLINE.md
5 langkah berikutnya:
1. Pantau run #1 (final/work/run1/lane_*.log); setelah selesai bandingkan submission_run1.csv dengan sub_v16dev (sha256 harus identik)
2. scripts/make_artifacts.sh final/work/run1 (model -> final/artifacts + SHA256SUMS); nama tim -> make_notebooks.py; scripts/run_notebook.sh untuk notebook penuh (final/nbrun_full) dan inference (final/nbrun_inf), keduanya harus memberi sha256 yang sama
3. Writeup: isi bagian 5-8 dan 11 (tabel model, hasil ablation, SHAP, kesimpulan), FINAL_REPORT.md
4. Final B: varian dari run yang sama (mis. tanpa koreksi kNN) atau v15nc; putuskan D-FINAL
5. Paket final + checklist NEEDS_USER paling lambat 23 Okt 12:00
Risiko / blocker / NEEDS_USER: TEAM_NAME = BISMILLAH (konfirmasi user); format writeup di guidebook Drive tidak terbaca
