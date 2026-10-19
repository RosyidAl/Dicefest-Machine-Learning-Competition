# STATUS (diperbarui: 2026-10-19 22:58 WIB)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya: arahan user 19 Okt 16:30: fokus submisi terbaik, notebook ditunda; boleh submit langsung (token Kaggle di file privat scratchpad, KAGGLE_API_TOKEN=<path>); submit hanya bila holdout/CV lebih baik dari v19
Fase selesai: Fase 0, Fase 1; FREEZE v16 (D006, tag freeze-v16); v17 Poisson ditolak (D007)
Insiden: 16 Okt ~20:40 WIB run #1 + E0260 kehabisan RAM (24 GB), container restart ~00:00. RAM sekarang 32 GB. 17 Okt 04:05 semua job nohup mati serentak.
Job berjalan: dice-run1s (logs/run1_shared.sh, run final v19 resume di final/work/run1): jalur CPU dilanjutkan otomatis setelah lp2_epa full selesai (sekitar 23:45; watcher menghentikan rantai v21 sebelum hold lalu SIGCONT 653822); jalur XGB + NN menunggu VRAM bebas >= 10 GB (proses luar PID host 931477 masih 13,2 GB). Setelah ketiga jalur rc=0 assemble otomatis dengan final/configs v19 dan cek sha 62444504 (uji drift); SESUDAH itu baru final/configs/resid_knn.json diganti ke exp/resid_knn_vc_AMp_e.json dan assemble ulang harus = sub_v19vce (38dd55b9). | Dev: dice-correst (PP0007 estimator alternatif koreksi kNN, exp/analysis/corr_est.log).
- Setelah run selesai: uji drift vs exp/oof|pred; make_artifacts + make_notebooks.py BISMILLAH + scripts/run_notebook.sh (full + inference). Notebook penuh butuh sekitar 23 GB disk lagi (bebas 11 GB) dan GPU kosong.
- Batas: server harus kembali paling lambat Selasa 20 Okt malam (run final sekitar 8 jam + notebook sekitar 30 jam)
Cek job: systemctl status dice-run1s. Selalu luncurkan job panjang dengan systemd-run (nohup setsid ikut terbunuh).
Terbaik CV_PRIMARY: v19 blend nested 0,24937 (5 target) | blend VC A2xM2xp2 (D012) nested 0,242107 untuk gas+elec+EUI (global 0,242231) | t4 0 | t5 0,0357
Holdout rantai penuh (blend VC A2xM2xp2): gas 0,24514, EUI 0,64878, elec 0,31462 (v19 global 0,24531 / 0,64930 / 0,31475). PP0006: EUI dengan konfigurasi kNN tambahan, cek jujur -0,000106
BEST_CURRENT.csv: sub_v19vce (38dd55b9...a9e6), LB publik 0,24923 (peringkat 6; #5 opo yo 0,24897, #4 Hbat 0,24809). sub_v19vc 0,24925, sub_v19dev 0,24934, sub_v19a3 0,24938 (D011). Kuota hari ini: 2 sisa (reset 07:00 WIB)
Figur Fase 12: reports/figures/fig1-fig5 (src/make_figures.py); outline presentasi: reports/PRESENTATION_OUTLINE.md
5 langkah berikutnya (fokus submisi terbaik, arahan user 19 Okt; notebook ditunda):
1. Selesai: sub_v19vce (D013). Konfigurasi final berikutnya = exp/v19vce (members v19 + exp/resid_knn_vc_AMp_e.json)
2. Anggota v21 ditolak untuk final (nested -0,000006 total); PP0007 estimator alternatif koreksi kNN: bila cek jujur negatif di kedua separuh, dukung konfigurasi (space, W, k, sp, est) di postprocess, rantai lambda, rakit, submit
3. Lanjutkan jalur CPU run final (kill -CONT 653822) setelah job LGB dev selesai; jalur GPU menunggu VRAM; lalu tambahkan VC + konfigurasi baru ke final/configs (members.yaml, resid_knn.json)
4. make_artifacts + notebook (butuh GPU kosong dan sekitar 23 GB disk); writeup/FINAL_REPORT/PRESENTATION: blend VC, LB 0,24925; D-FINAL
5. Paket final + checklist NEEDS_USER paling lambat 23 Okt 12:00
Risiko / blocker / NEEDS_USER: GPU dipakai proses luar container (PID host 931477, 13,2 GB) sejak 19 Okt siang: run final jalur GPU dan notebook penuh tertahan (perlu sekitar 30 jam server sebelum 23 Okt); disk bebas 9 GB (notebook butuh sekitar 23 GB); TEAM_NAME = BISMILLAH; format writeup di guidebook Drive tidak terbaca

## HANDOFF (18 Okt 18:35, sebelum migrasi akun Claude; konteks lengkap di HANDOVER.md)
- Tidak ada job DICE FEST yang berjalan. Server dipakai user untuk proyek /root/kaggle; jangan jalankan apa pun sebelum user mengizinkan.
- Model final: v19 (D010). sub_v19dev public LB 0,24934 (v18 0,24936, v16 0,24943). CV nested 0,24937. Konfigurasi final sudah v19: final/configs/members.yaml (40 anggota) + resid_knn.json (λ v19). Cadangan: members_v18.yaml / resid_knn_v18.json, members_v16.yaml / resid_knn_v16.json.
- Sisa pekerjaan yang butuh server (sekitar 21 jam; mulai paling lambat Selasa 20 Okt malam):
  1. Run final v19: `systemd-run --unit=dice-run1 --working-directory=/root/Dicefest -E HOME=/root -p StandardOutput=append:/root/Dicefest/logs/run1.log -p StandardError=append:/root/Dicefest/logs/run1.log .venv/bin/python -u final/run_all.py --work final/work/run1 --out final/output/submission_run1_v19.csv` (resume; melatih lk2_gpm, lp_gpm, lp_epm, xp82_gpm, xp2_gpgea, xp2_eui, xp2_epa, lp2_gpm, lp2_epm, pk2_lgpm; sekitar 4-5 jam).
  2. Uji drift: array baru di final/work/run1/{oof,pred} harus identik byte dengan exp/{oof,pred}; sha256 submission_run1_v19.csv harus = sub_v19dev (62444504fb7de677d10fd69d17b06da796076b75e96d103cef2d22dd95dff216).
  3. `bash scripts/make_artifacts.sh final/work/run1` (model -> final/artifacts + SHA256SUMS), `.venv/bin/python final/make_notebooks.py BISMILLAH`, lalu `bash scripts/run_notebook.sh final/DICEFEST2026_BISMILLAH.ipynb final/nbrun_full` (sekitar 16 jam) dan `bash scripts/run_notebook.sh final/inference_from_artifacts.ipynb final/nbrun_inf`; keduanya harus memberi sha256 yang sama.
  4. Writeup PDF (scripts/build_writeup_pdf.py dari venv terpisah: python3 -m venv pdfenv && pdfenv/bin/pip install xhtml2pdf markdown-it-py pymupdf), isi angka akhir (v19, sha256, waktu jalan) di reports/WRITEUP_DRAFT.md, FINAL_REPORT.md, PRESENTATION_OUTLINE.md; D-FINAL + checklist NEEDS_USER.
- Pelajaran teknis: (a) SIGSTOP tidak membebaskan RAM/VRAM; cek RAM dan VRAM bebas sebelum meluncurkan job GPU. (b) Tiga jalur run_all yang menghitung FB07b bersamaan memakan sampai 29 GB; final/src/fb07b.py kini THREADS=3. (c) OpenBLAS multithread mengubah bit terakhir: assemble memakai threadpool_limits(1) (D008). (d) Pemantau latar Claude Code bisa dihentikan saat RAM menipis; job systemd tetap jalan.
