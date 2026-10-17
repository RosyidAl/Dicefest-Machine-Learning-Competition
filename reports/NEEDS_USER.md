# NEEDS_USER (hal yang butuh tindakan user)

- [x] (16 Okt) BEST_CURRENT (sub_v15ncrkd) sudah di-submit user: public LB 0,24954.
- [x] (17 Okt) BEST_CURRENT (sub_v16dev) sudah di-submit user: public LB 0,24943 (v15ncrkd 0,24954).
- [ ] Kredensial Kaggle CLI belum ada (KAGGLE_AUTO_SUBMIT: false), jadi semua submit dilakukan manual oleh user.
- [ ] TEAM_NAME masih "ISI_NAMA_TIM". Nama tim dibutuhkan untuk nama notebook final `final/DICEFEST2026_<TEAM_NAME>.ipynb`.
- [ ] GPU dipakai bersama container lain (sekitar 11 GB VRAM terpakai dari luar). Bila memungkinkan, minta admin memastikan GPU cukup longgar selama run audit final (17 sampai 21 Okt).
- [ ] Ketentuan format writeup PDF dan aturan penamaan file ada di guidebook lengkap (folder Google Drive panitia, lewat dicefest.web.id/guidebook/machine-learning) yang tidak bisa diakses dari server. Mohon cek: batas halaman, template, dan nama file notebook/PDF. Draf saat ini ada di reports/WRITEUP_DRAFT.md dan mengikuti struktur MASTER_PROMPT.
- [ ] (17 Okt 13:50) Disk: 26 GB bebas. Run #1 butuh sekitar 8 GB lagi dan run notebook kedua sekitar 20 GB. Minta izin membersihkan ~/.cache/pip (1,6 GB) dan ~/.cache/uv (sekitar 6,4 GB riil) bila kurang. ~/.cache/huggingface (12 GB, model audio burung, proyek lain) tidak disentuh.
