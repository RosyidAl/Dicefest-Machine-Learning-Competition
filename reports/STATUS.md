# STATUS (diperbarui: 2026-10-19 13:10 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: v18 (arahan user: hasil terbaik) + run #1 final v16 di latar / seleksi v18 18 Okt
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan: TIDAK ADA. 19 Okt 13:0x user meminta semua job DICE FEST dihentikan (jalur CPU run final v19 sempat berjalan sekitar 10 menit lalu dihentikan; belum ada anggota baru yang selesai). Jangan jalankan apa pun sebelum user mengizinkan.
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

## HANDOFF (18 Okt 18:35, sebelum migrasi akun Claude; konteks lengkap di HANDOVER.md)
- Tidak ada job DICE FEST yang berjalan. Server dipakai user untuk proyek /root/kaggle; jangan jalankan apa pun sebelum user mengizinkan.
- Model final: v19 (D010). sub_v19dev public LB 0,24934 (v18 0,24936, v16 0,24943). CV nested 0,24937. Konfigurasi final sudah v19: final/configs/members.yaml (40 anggota) + resid_knn.json (λ v19). Cadangan: members_v18.yaml / resid_knn_v18.json, members_v16.yaml / resid_knn_v16.json.
- Sisa pekerjaan yang butuh server (sekitar 21 jam; mulai paling lambat Selasa 20 Okt malam):
  1. Run final v19: `systemd-run --unit=dice-run1 --working-directory=/root/Dicefest -E HOME=/root -p StandardOutput=append:/root/Dicefest/logs/run1.log -p StandardError=append:/root/Dicefest/logs/run1.log .venv/bin/python -u final/run_all.py --work final/work/run1 --out final/output/submission_run1_v19.csv` (resume; melatih lk2_gpm, lp_gpm, lp_epm, xp82_gpm, xp2_gpgea, xp2_eui, xp2_epa, lp2_gpm, lp2_epm, pk2_lgpm; sekitar 4-5 jam).
  2. Uji drift: array baru di final/work/run1/{oof,pred} harus identik byte dengan exp/{oof,pred}; sha256 submission_run1_v19.csv harus = sub_v19dev (62444504fb7de677d10fd69d17b06da796076b75e96d103cef2d22dd95dff216).
  3. `bash scripts/make_artifacts.sh final/work/run1` (model -> final/artifacts + SHA256SUMS), `.venv/bin/python final/make_notebooks.py BISMILLAH`, lalu `bash scripts/run_notebook.sh final/DICEFEST2026_BISMILLAH.ipynb final/nbrun_full` (sekitar 16 jam) dan `bash scripts/run_notebook.sh final/inference_from_artifacts.ipynb final/nbrun_inf`; keduanya harus memberi sha256 yang sama.
  4. Writeup PDF (scripts/build_writeup_pdf.py dari venv terpisah: python3 -m venv pdfenv && pdfenv/bin/pip install xhtml2pdf markdown-it-py pymupdf), isi angka akhir (v19, sha256, waktu jalan) di reports/WRITEUP_DRAFT.md, FINAL_REPORT.md, PRESENTATION_OUTLINE.md; D-FINAL + checklist NEEDS_USER.
- Pelajaran teknis: (a) SIGSTOP tidak membebaskan RAM/VRAM; cek RAM dan VRAM bebas sebelum meluncurkan job GPU. (b) Tiga jalur run_all yang menghitung FB07b bersamaan memakan sampai 29 GB; final/src/fb07b.py kini THREADS=3. (c) OpenBLAS multithread mengubah bit terakhir: assemble memakai threadpool_limits(1) (D008). (d) Pemantau latar Claude Code bisa dihentikan saat RAM menipis; job systemd tetap jalan.
