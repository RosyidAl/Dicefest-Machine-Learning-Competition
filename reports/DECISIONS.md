# DECISIONS (format ADR)

### D000 [2026-10-16 11:28 WIB] Menerapkan MASTER_PROMPT di atas pekerjaan yang sudah berjalan
- Konteks: MASTER_PROMPT baru diberikan 16 Okt, setelah 4 hari pemodelan (12 sampai 16 Okt).
  Yang sudah ada: 168 model di exp/oof + exp/pred, 17 versi blend, public LB terbaik 0,2499 (sub_v15ncrd).
  Struktur kode lama ada di src/ (train_lib, nn, te, prune, resid_corr, resid_knn) dan exp/.
- Opsi:
  - (a) mulai ulang dari nol sesuai struktur MASTER_PROMPT;
  - (b) pertahankan kode dan artefak lama, backfill ke registry, lalu lanjutkan fase dengan struktur baru.
- Bukti: run ulang model lama identik byte (XGB, LGB, MLP), jadi artefak lama bisa direproduksi.
  Mengulang semuanya butuh sekitar 28 jam GPU + CPU hanya untuk kembali ke titik yang sama.
- Keputusan: (b).
  - Run lama tercatat sebagai E0001 sampai E0168 (model), ENS0001 sampai ENS0017 (blend), PP0001 sampai PP0004 (post-processing).
  - Feature set lama dinamai FS_legacy.
  - Eksperimen baru memakai registry, configs/ dan experiments/<id>/.
- Risiko dan mitigasi: run lama tidak punya folder experiments/<id>/ lengkap. Mitigasinya, log, OOF dan skrip sumber dirujuk di kolom notes/config_path.

### D001 [2026-10-16 11:28 WIB] Mekanisme split train/test: acak per baris
- Bukti awal (EDA 12 Okt, diverifikasi ulang di Fase 1, angka lengkap di reports/eda/EDA_REPORT.md):
  - 429.303 / 613.290 = 0,7000 tepat;
  - semua 330 zone_region_id test ada di train;
  - sekitar 92% subzone test ada di train;
  - urutan id acak;
  - CV KFold per baris cocok dengan public LB di 9 submisi.
- Keputusan: test = sampel acak 30% per baris. Informasi tingkat grup (TE subzone/zone, agregat) sah dan dipakai.

### D002 [2026-10-16 11:28 WIB] CV_PRIMARY = KFold(shuffle=True, random_state=42) per baris, 10-fold untuk model final
- Konteks: MASTER menyarankan StratifiedKFold pada 20 bin t1 untuk split acak.
  Semua 168 model lama memakai KFold(5 atau 10, shuffle, 42) per baris.
- Bukti: CV vs public LB konsisten.
  - v8 0,25061 ke 0,25116; v9r ke 0,25070; v12r 0,25015 ke 0,25025; v13ra ke 0,25008; v15rd 0,25001 ke 0,25000.
  - Offset stabil, urutan selalu benar.
- Keputusan: pertahankan KFold. File permanen: data/folds/folds_primary_k5_s42.parquet dan folds_primary_k10_s42.parquet.
  tests/test_folds.py membuktikan keduanya identik dengan split train_lib.
  CV_SECONDARY: data/folds/folds_secondary_k5_s42.parquet (zone dialokasikan acak ber-seed, baris seimbang).
- Risiko: blend lama mencampur anggota 5-fold dan 10-fold (pitfall bagian 18). Tiap OOF tetap jujur out-of-fold.
  Bobot NNLS dinilai dengan nested CV, dan holdout P70/H30 memberi bobot setara (0,25095 vs 0,25096).
  Dampaknya kecil dan tercatat.

### D003 [2026-10-16 11:28 WIB] Gerbang freeze dimajukan
- Konteks: pipeline final (50 anggota blend v15nc) butuh sekitar 19 jam GPU + 9 jam CPU (paralel), jadi sekitar 20 jam per run.
  Audit butuh 3 run penuh (run #1, run #2, notebook headless), sekitar 60 jam berurutan.
  GPU juga dipakai bersama container lain.
- Keputusan: freeze kandidat final paling lambat 18 Okt 18:00 WIB (bukan 22 Okt 12:00).
  Run #1 dimulai segera setelah freeze, disusul run #2 dan notebook.
  Eksperimen yang selesai setelah freeze hanya masuk bila ada waktu untuk mengulang ketiga run.
- Risiko: perbaikan setelah freeze tidak ikut final. Mitigasi: prioritaskan eksperimen bernilai tinggi sebelum freeze.

### D004 [2026-10-16 11:47 WIB] Determinisme per keluarga model (Fase 2, 6.4)
- Bukti:
  - LightGBM CPU (deterministic=True, force_row_wise, num_threads tetap): rerun lgb_epm_wM_tew identik byte (14 Okt).
  - XGBoost CUDA hist: rerun anggota zoo1/zoo3/zoo5/grid identik byte (14 Okt).
  - CatBoost GPU: TIDAK identik. Rerun c4_lepa memberi iterasi early-stopping berbeda dan rms beda prediksi test 352 kWh.
  - MLP polos PyTorch (cudnn deterministic, use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG): rerun n4f_lepm identik byte.
  - MLP PLR, TabM-mini + PLR, multi-task (R05, 16 Okt, scripts/check_repro_nn.py + check_repro.sh, 2 proses terpisah):
    hash prediksi identik untuk keempat varian (logs/repro_check_repro_nn.result).
- Keputusan: pipeline final hanya memakai LightGBM, XGBoost, dan PyTorch (blend "nc", tanpa CatBoost).
  Biaya membuang CatBoost: CV +0,00001 (v15 0,25001 vs v15nc 0,25002). Di public LB, v15ncrd (0,2499) malah lebih baik dari v15rd (0,25000).
- Risiko: GPU dipakai bersama container lain. Peringatan OOM sempat muncul saat prediksi TabM (alokasi 1,25 GB gagal lalu
  dicoba ulang), tetapi hasilnya tetap identik. Mitigasi: chunk prediksi TabM 1024 baris; jalur GPU final hanya satu proses.

### D005 [2026-10-16 13:04 WIB] Komposisi kandidat final
- Konteks: FB07 DITERIMA. Pruning menunjukkan 17 anggota v15nc bisa dibuang tanpa rugi.
- Opsi:
  - (a) v15nc apa adanya;
  - (b) v15nc pruned + anggota FB07, dengan bobot koreksi residual di-fit ulang di holdout yang memuat anggota FB07.
- Bukti:
  - E0191 vs E0190: gas -0,0011 dan EUI -0,0022 model tunggal;
  - runtime_prune2.log: +0,000003 untuk 9,26 jam lebih cepat.
- Keputusan: (b) jika holdout membuktikan rantai blend + koreksi baru lebih baik. Bila tidak, (a) dengan pruning.
  Freeze setelah λ baru di-fit (target 17 Okt siang), lalu run #1.
- Risiko: λ koreksi lama di-fit tanpa fitur kNN di model; dipakai apa adanya bisa over-correct. Mitigasi: fit ulang wajib.
