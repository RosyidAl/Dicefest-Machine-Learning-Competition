# NEEDS_USER (hal yang butuh tindakan user)

- [x] (16 Okt) BEST_CURRENT (sub_v15ncrkd) sudah di-submit user: public LB 0,24954.
- [x] (17 Okt) BEST_CURRENT (sub_v16dev) sudah di-submit user: public LB 0,24943 (v15ncrkd 0,24954).
- [x] (19 Okt) Token Kaggle diberikan user; submit dilakukan langsung dari server (token di file privat scratchpad, tidak masuk repo), hanya untuk kandidat yang lebih baik di rantai holdout.
- [x] (17 Okt) TEAM_NAME = BISMILLAH (konfirmasi user).
- [ ] (19 Okt, PENTING) Sejak 19 Okt siang GPU dipakai proses di luar container (PID host 931477, 13,2 GB VRAM, utilisasi 100%) ditambah job kaggle user (2,3 GB). Jalur GPU run final (XGB + NN) dan notebook penuh tidak bisa jalan sampai VRAM bebas >= 10 GB. Mohon cek ke admin server siapa pemilik proses itu dan kapan selesai. Sebelumnya: GPU dipakai bersama container lain (sekitar 11 GB VRAM terpakai dari luar). Bila memungkinkan, minta admin memastikan GPU cukup longgar selama run audit final (17 sampai 21 Okt).
- [ ] Ketentuan format writeup PDF dan aturan penamaan file ada di guidebook lengkap (folder Google Drive panitia, lewat dicefest.web.id/guidebook/machine-learning) yang tidak bisa diakses dari server. Mohon cek: batas halaman, template, dan nama file notebook/PDF. Draf saat ini ada di reports/WRITEUP_DRAFT.md dan mengikuti struktur MASTER_PROMPT.
- [x] (17 Okt 13:50, cache pip/uv dihapus atas izin user, 28 GB bebas) Disk: 26 GB bebas. Run #1 butuh sekitar 8 GB lagi dan run notebook kedua sekitar 20 GB. Minta izin membersihkan ~/.cache/pip (1,6 GB) dan ~/.cache/uv (sekitar 6,4 GB riil) bila kurang. ~/.cache/huggingface (12 GB, model audio burung, proyek lain) tidak disentuh.
- [x] (17 Okt 14:00, dijeda SIGSTOP atas izin user; lanjutkan: kill -CONT 141266) Job lain (`/root/kaggle`, train_noisy_student.py) memakai GPU yang sama (GPU 100%, VRAM 11,9/16,4 GB). Run #1 melambat dan ada risiko OOM di jalur NN. Mohon tunda job itu sampai run final + notebook DICE FEST selesai, atau terima perlambatan (lane yang gagal bisa di-resume).

## Checklist akhir untuk user (draf, dilengkapi setelah run audit)
1. Kaggle: pilih submisi final A = output notebook final. Bila sha256 run #1 dan notebook sama dengan sub_v16dev (e9375ba6...9a7f), cukup pilih submisi sub_v16dev yang sudah ada (public LB 0,24943). Final B: lihat D-FINAL.
2. Website dicefest.web.id (paling lambat 23 Okt 23:59 WIB): unggah `final/DICEFEST2026_<TEAM>.ipynb` (versi tereksekusi) dan PDF writeup (`reports/WRITEUP.pdf`, dibangun dengan `scripts/build_writeup_pdf.py`).
3. Cek ketentuan format writeup (batas halaman, template, nama file) di guidebook, lalu kabari bila perlu penyesuaian.
4. Simpan `final/artifacts/` (model + SHA256SUMS) untuk audit 24 Okt; notebook cadangan `final/inference_from_artifacts.ipynb` memakai folder ini.

## Langkah akhir untuk user (22 Okt)
- [ ] Kaggle: pilih submisi final `sub_final_pipeline.csv` (LB publik 0,24913, sha aec745ab). Sebaiknya hanya file ini, karena hanya file ini yang direproduksi persis oleh notebook (D-FINAL).
- [ ] Website dicefest.web.id (paling lambat 23 Okt 23:59 WIB): unggah `final/deliverables/DICEFEST2026_BISMILLAH.ipynb` (sudah tereksekusi, dengan output) dan `final/deliverables/WRITEUP_BISMILLAH.pdf` (10 halaman).
- [ ] Cek guidebook: aturan nama file, batas halaman PDF, dan apakah folder model tersimpan (sekitar 27 GB, untuk notebook inference) boleh dilampirkan lewat tautan.
