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

### D006 [2026-10-16 20:24 WIB] FREEZE kandidat final v16
- Konteks: semua anggota FB07 selesai (full + holdout); seleksi dan fit ulang λ selesai.
- Bukti:
  - Seleksi (exp/analysis/select_v16.log): v15nc dikurangi 17 anggota hasil pruning, ditambah forward-add anggota FB07 lalu backward-drop.
    Hasilnya 34 anggota (22 gas, 12 elec), blend honest 0,242467 vs 0,242886 untuk v15nc.
  - Holdout rantai penuh (exp/resid_knn_fit_v16.log), v15 → v16: gas 0,24592 → 0,24539; EUI 0,65087 → 0,64961; elec 0,31569 → 0,31477.
    Total sekitar -0,00054; ekspektasi LB sekitar 0,2490.
- Keputusan:
  - Final A = v16 (final/configs/members.yaml + resid_knn.json = exp/resid_knn_v16.json).
  - Final B (lindung nilai) = v15nc-pruned tanpa FB07 (members_v15nc.yaml, λ v15), komposisi berbeda dengan CV sedikit lebih buruk; dibangun bila waktu GPU tersisa.
  - Kode final/ dibekukan (git tag freeze-v16). Setelah ini hanya perbaikan bug.
- Risiko: holdout memakai semua anggota holdout, bukan persis 34 anggota final (approksimasi yang sama seperti v15, terbukti akurat di LB).

### D007 [2026-10-17 11:15 WIB] Anggota Poisson (v17) tidak masuk final; v16 tetap Final A
- Konteks: A/B Poisson menang besar di model tunggal (E0260-E0265), lalu 4 anggota XGB Poisson penuh dilatih (xp_gpm, xp_eui, xp_epm, xp_gpgea).
- Bukti:
  - Blend honest nested (exp/analysis/select_v17_r3.log): 0,242467 -> sekitar 0,242432 (-0,000035), walau model tunggal xp_gpm adalah anggota EUI terbaik (0,65741).
  - Holdout rantai penuh (blend + koreksi grup/kNN, semua anggota holdout termasuk xp_gpm/xp_eui/xp_epm): gas 0,24539 -> 0,24538, EUI 0,64961 -> 0,64959. Praktis nol.
  - MASTER_PROMPT bagian 12 (Fase 8): gain di bawah 0,1% relatif (sekitar 0,00025) -> pilih yang lebih sederhana.
- Keputusan: v16 tetap Final A (members.yaml dan resid_knn.json tidak diubah). Rantai v17 (xp_epa, xp8_gpm, lp_*) dihentikan, GPU dikembalikan ke run #1 (SIGCONT 11:11).
- Risiko: tidak ada; pipeline final tidak berubah.

### D008 [2026-10-17 22:58 WIB] Perbaikan determinisme tahap assemble (BLAS satu thread)
- Konteks: run #1 mereproduksi semua array anggota byte per byte, tetapi file submisi berbeda di bit terakhir pada 24 nilai karena OpenBLAS gemv multithread.
- Keputusan: assemble dan koreksi residual dijalankan dengan threadpool_limits(1). Ini perbaikan bug (diizinkan setelah freeze D006), tidak mengubah model.
- Risiko: tidak ada pada skor (selisih relatif <= 4e-16). File final harus berasal dari kode yang sudah diperbaiki; sub_v16dev di Kaggle tidak identik byte dengan output notebook.

### D009 [2026-10-18 02:41 WIB] v18 menggantikan v16 sebagai kandidat Final A (meninjau D007)
- Konteks: user meminta hasil terbaik walau gain kecil (17-18 Okt). D007 menolak anggota Poisson saja (-0,000035) dengan aturan 0,1%.
- Bukti: seleksi nested v18 (FB07b + Poisson, 38 anggota) -0,000202; rantai holdout membaik di ketiga metrik (gas -0,00006, EUI -0,00024, elec -0,00001; total -0,00006).
- Keputusan: v18 = Final A setelah dirakit dan dicek di LB. Pipeline final perlu diperbarui: anggota FB07b (fb07b.DataX) dan Poisson masuk members.yaml, resid_knn.json = exp/resid_knn_v18.json, lalu uji drift (array final identik dengan array pengembangan).
- Risiko: waktu jalan pipeline bertambah (sekitar 14 anggota baru); kode baru (fb07b.py) harus masuk final/src dan notebook. Final B = v16 (sub_v16fin, pipeline sudah terbukti) sebagai lindung nilai.

### D010 [2026-10-18 11:42 WIB] v19 menggantikan v18 sebagai kandidat Final A
- Bukti: seleksi nested -0,000034 di atas v18; rantai holdout membaik di ketiga metrik (total sekitar -0,00002).
- Keputusan: v19 = Final A (arahan user: maksimal); Final B = v16 (sub_v16fin, pipeline final sudah terbukti end-to-end) atau v18 sebagai lindung nilai, diputuskan di D-FINAL.
- Risiko: 7 anggota tambahan memperpanjang notebook sekitar 3-4 jam; semua anggota baru harus lolos uji drift (array final identik dengan dev).

### D011 [2026-10-19 17:55 WIB] Adopsi v3 (PP0005) ditolak, adopt_v2 dipertahankan
- Bukti CV (5-fold subzona, bobot campuran baris subzona test baru): rata-rata 8 model LightGBM (base, census-kNN, bobot campuran) 1,0283 vs adopt_v2 1,0303, perkiraan -0,000016 total.
- Bukti LB publik: sub_v19a3 (hanya kolom adopsi subzona baru yang berbeda) 0,24938 vs sub_v19dev 0,24934. Simulasi subset publik dari galat CV memberi selisih -0,000016 (sd 0,000017); peluang hasil >= +0,00003 sekitar 0,3%.
- Keputusan: anggap ada perbedaan sistematis antara subzona test baru dan proksi CV-nya; adopt_v2 tetap dipakai. Tidak ada lagi tuning adopsi berdasarkan CV saja.
- Risiko: kecil; perubahan yang ditolak bernilai sekitar 0,00002 menurut CV.

### D012 [2026-10-19 18:55 WIB] Blend dengan bobot yang bervariasi (varying-coefficient) menggantikan blend NNLS global
- Bukti nested 5-fold jujur (anggota v19): A2 (peringkat luas, 2 knot) -0,000039; AxM -0,000077; level prediksi -0,000067; A2xM2xp2 -0,000124 (EUI 0,650288 -> 0,649839); spesifikasi yang lebih kaya (A3xM2xp2, A2xM2xp3) tidak lebih baik.
- Bukti rantai holdout (src/chain_vc.py, λ di-fit ulang): A2xM2xp2 gas 0,24514 / EUI 0,64878 / elec 0,31462 vs v19 0,24531 / 0,64930 / 0,31475 (total -0,000164); A2 saja -0,000058.
- Bukti LB publik: sub_v19vc 0,24925 vs sub_v19dev 0,24934 (-0,00009).
- Keputusan: sub_v19vc = BEST_CURRENT dan kandidat Final A baru. Bobot anggota = interpolasi multilinear (hat basis) pada peringkat A (bld_dim_06), M (bld_dim_12) dan prediksi blend global p; semuanya fungsi fitur/prediksi, bukan target. Implementasi: blend.blend_vc, postprocess.blend_preds(spec), pipeline._assemble membaca "vc" dari resid_knn.json.
- Risiko: konfigurasi final harus memuat exp/resid_knn_vc_AMp.json (λ + vc); uji regresi jalur global (v19 harus tetap 62444504...) belum selesai (dihentikan karena RAM), dijadwalkan ulang.

### D013 [2026-10-19 22:53 WIB] Konfigurasi koreksi kNN EUI diperluas (PP0006); gas dan elec tetap
- Bukti: seleksi dari nol di seluruh H tampak lebih baik di ketiga metrik, tapi cek jujur split-half (pilih di separuh H, nilai di separuh lain) memberi gas +0,000054, EUI -0,000059, elec +0,000019. Varian "CONFIGS sekarang + tambahan serakah": gas +0,000003, EUI -0,000106 (kedua separuh negatif), elec +0,000023.
- Rantai holdout dengan 7 konfigurasi EUI tambahan (exp/chain_vc_AMp_e.log): EUI 0,64878 -> 0,64859; gas dan elec identik (λ sama persis).
- LB publik: sub_v19vce 0,24923 vs sub_v19vc 0,24925; kolom gas, elec, rasio bersih dan adopsi identik byte dengan sub_v19vc.
- Keputusan: sub_v19vce = BEST_CURRENT dan kandidat Final A. Konfigurasi kNN kini dibaca dari resid_knn.json ("configs") di pipeline final; konfigurasi final = exp/resid_knn_vc_AMp_e.json.
