# JOURNAL

### [2026-10-12 s/d 2026-10-16 WIB] E0001-E0168, ENS0001-ENS0017, PP0001-PP0004 | Fase 1-9 (pra-registry) | ringkasan pekerjaan lama
- Hipotesis utama yang diuji: identitas fisik antar target, TE spasial, formulasi rasio terhadap exposure, keragaman model untuk blend, koreksi residual grup.
- Hasil:
  - identitas eksak: clean = round(100 b02/b01, 2), EUI = gas / bld_dim_06, adoption konstan per subzone;
  - TE subzone/zone/window/sel iklim; agregat transductive;
  - 168 model (XGB, LGB, CatBoost, MLP, PLR, TabM-mini, multi-task);
  - blend NNLS honest 0,25065 (v6) turun ke 0,25001 (v15);
  - koreksi residual grup dan kNN, adopt_v2, dup_fix.
  - Public LB: 0,25317, lalu 0,25125, 0,25116, 0,25070, 0,25025, 0,25008, 0,25000, dan 0,2499.
- Kesimpulan: blend jenuh (anggota baru memberi gain <= 0,00004). Gain terakhir datang dari post-processing residual.
- Tindak lanjut: struktur MASTER_PROMPT (D000), coverage pendekatan yang belum dicoba, pipeline final reproducible.

### [2026-10-16 11:28 WIB] Fase 0 | setup lingkungan dan data
- Setup:
  - env/ENVIRONMENT.md, requirements.lock;
  - smoke test GPU (XGB CUDA, CatBoost GPU, torch CUDA, LGB, FAISS GPU semua OK);
  - data/raw (hardlink read-only), verifikasi md5/shape/prefix, cache parquet identik;
  - src/metric.py (resmi verbatim), seed.py, folds.py, tracking.py, scripts/validate_submission.py, update_reports.py;
  - tests (6 lulus), fold permanen, git init.
- Paket opsional terpasang: tabm, rtdl_num_embeddings, gpboost, umap-learn, pytabkit, faiss-gpu-cu12.
  cuml tidak dipasang karena dry-run akan menurunkan numpy.
- Kesimpulan: Fase 0 selesai. Registry di-backfill dengan 168 model + 17 blend + 4 post-processing.

### [2026-10-16 11:58 WIB] Fase 1 | EDA dan forensik split
- Hipotesis: H-SPLIT (split acak per baris), H-FISIKA (gas = EUI x luas, t5 terkait hvac_dist).
- Setup: src/eda.py (integritas, forensik split, adversarial LGB 5-fold dengan/tanpa ID, target, fitur, struktur fisik).
- Hasil:
  - H-SPLIT DITERIMA: unseen-subzone test share 8,14% vs 8,17% harapan; AUC 0,4997/0,4990.
  - H-FISIKA DITERIMA: identitas eksak gas = EUI x bld_dim_06; clean eksak; adopsi = sisa hvac - kategori hilang, R2 0,978.
- Tindak lanjut: tidak ada ide baru. Semua implikasi sudah terimplementasi di pipeline (TE grup, formulasi rasio, dup_fix, adopt_v2).

### [2026-10-16 11:58 WIB] Fase 2-11 (paralel) | pipeline final dan uji
- final/src (modul beku), final/configs/members.yaml (50 anggota v15nc persis dari skrip asal), final/run_all.py, final/src/pipeline.py, final/src/inference.py, final/make_notebooks.py.
- Uji assemble dengan array lama: sub identik dengan sub_v15ncrkd sampai 2e-16 relatif (round-trip CSV). Honest CV blend 0,24591 / 0,65211 / 0,31641 (cocok dengan prune_v15nc.log).
- Tes kebocoran (tests/test_no_leakage.py): TE grup dan FB07 knn_target lulus.

### [2026-10-16 12:36 WIB] Fase 11 (awal) | uji drift pipeline final
- Hipotesis: final/run_all.py (salinan beku + members.yaml) mereproduksi array OOF/test pengembangan bit per bit.
- Setup: --stage members --only {xgb_lepm, xgb_eui_d10, xgb_eui_wA_te, n4f_gpm} (GPU) dan lgb_epm_wM_tew (CPU), lalu bandingkan dengan exp/oof dan exp/pred.
- Hasil: kelima anggota identik byte (oof dan pred). Ukuran model sekitar 100 MB per anggota 5-fold.
- Kesimpulan: DITERIMA. Pipeline final setara dengan kode pengembangan, dan spesifikasi members.yaml benar untuk keluarga XGB (dengan/tanpa TE), MLP, dan LGB.

### [2026-10-16 13:04 WIB] E0190-E0191 | Fase 4 | FB07 KNN-target OOF
- Hipotesis: rata-rata target (log EUI, log elec/meter, log gas/meter) dari k baris train paling mirip stok bangunannya dalam window +-W sub_n, cross-fitted, menurunkan error model.
  Dasar: sinyal ini sudah terbukti lewat koreksi residual kNN (PP0004).
- Setup:
  - XGB gpm (x4 params, lr 0,05, d8), FS_legacy + agg + TE, 5-fold primary;
  - E0191 menambah blok knn_target: W/k = 30/20, 100/20 (spasial 4), 300/50; inner 5-fold untuk baris training;
  - tes kebocoran LK1 lulus.
- Hasil:

  | Target | E0190 | E0191 | Delta | Fold membaik | SE |
  |---|---|---|---|---|---|
  | gas | 0,25291 | 0,25181 | -0,00111 | 5/5 | 0,00008 |
  | EUI | 0,66444 | 0,66229 | -0,00215 | 5/5 | 0,00064 |

- Kesimpulan: DITERIMA (memenuhi kriteria 6.2).
- Tindak lanjut:
  - anggota baru xk_gpm, xk_lepa (XGB), lk_gpm, lk_lepa (LGB), pk_gpm (PLR MLP), full 10-fold + holdout P70/H30 (src/exp_v16_knn.py);
  - lalu fit ulang bobot koreksi residual di holdout dan prune blend.

### [2026-10-16 13:04 WIB] Analisis | pruning anggota sadar runtime (exp/analysis/runtime_prune*.log)
- Hipotesis: sebagian anggota v15nc berbobot hampir nol dan bisa dibuang tanpa kehilangan skor, untuk menghemat waktu tiga run audit.
- Hasil:
  - membuang 17 anggota (urutan rasio rugi/jam) menghemat 9,26 jam member-time dengan perubahan honest CV +0,000003 (noise);
  - langkah berikutnya mulai rugi: t1_lepm +0,00002 per 2,9 jam.
- Kesimpulan: dipakai saat menyusun daftar anggota final (D005).

### [2026-10-16 14:36 WIB] EM0001 (xk_gpm) | Fase 4/9 | gain bersih FB07 setelah koreksi residual (holdout)
- Hipotesis: anggota FB07 menurunkan error rantai penuh (blend + koreksi residual grup/kNN dengan λ di-fit ulang), bukan hanya blend mentah.
- Setup: xk_gpm full (10-fold) dan holdout (P70/H30); src/resid_knn.py fit pada semua anggota holdout + xk_gpm → exp/resid_knn_v16a.json.
- Hasil:
  - Model tunggal xk_gpm: EUI 0,65977 (terbaik), gas 0,25042.
  - Blend OOF honest (v15nc + xk_gpm): -0,000174 total.
  - Holdout rantai penuh, gas: H 0,24735 → 0,24680; setelah koreksi 0,24592 → 0,24569.
  - Holdout rantai penuh, EUI: H 0,65308 → 0,65178; setelah koreksi 0,65087 → 0,65010.
  - Total sekitar -0,0002.
- Kesimpulan: DITERIMA. Tambah xk_lgpm, xk_gpgea (gas), pk_lepm (elec); xk_lepa, lk_gpm, lk_lepa, pk_gpm sedang berjalan.
- LB: BEST_CURRENT (sub_v15ncrkd) = 0,24954 (user, 16 Okt), sesuai perkiraan holdout.

### [2026-10-16 20:39 WIB] E0200-E0208 | Fase 3 | screening transform target (LGB lr 0,1, 3 fold, FS_legacy + agg + TE)
- Hipotesis: transform T1 sampai T6 menurunkan NRMSE dibanding identity (E0200).
- Hasil (delta vs identity, fold membaik):

  | Transform | gas | elec | EUI |
  |---|---|---|---|
  | T1 log1p | +0,0017 (0/3) | +0,0023 | +0,0030 |
  | T2 sqrt | -0,0006 (3/3) | -0,0005 | 0,0000 |
  | T3 Yeo-Johnson | +0,0006 | +0,0015 | +0,0009 |
  | T4 rasio/meter E^2 | -0,0010 (3/3) | -0,0016 | n/a |
  | T5 Tweedie 1,5 | -0,0009 | -0,0004 | -0,0005 |
  | T5 Gamma | +0,0008 | +0,0015 | 0,0000 |
  | **T5 Poisson** | **-0,0041 (3/3)** | **-0,0029** | **-0,0049** |
  | T6 Huber log | -0,0009 | +0,0002 | -0,0008 |

- Kesimpulan:
  - Poisson paling kuat, juga dibanding T4 (gas 0,25467 vs 0,25775).
  - Masuk akal: noise label = rata-rata rumah tangga, sehingga varians sebanding dengan mean.
  - log1p tanpa koreksi bias kalah; anggota log kita tetap berguna di blend karena bobot NNLS menyerap bias.
- Tindak lanjut: A/B Poisson vs formulasi rasio terbaik (E0260 sampai E0263). Bila menang, buat anggota Poisson (+ holdout) dan pertimbangkan freeze v17.

### [2026-10-17 00:45 WIB] Insiden OOM dan peluncuran ulang run #1 | Fase 11
- Kejadian: run #1 final v16 (3 jalur, mulai 20:25 WIB) berjalan bersamaan dengan E0260 (LGB) di RAM 24 GB.
  Sekitar 20:40 WIB memori habis; E0260 butuh 3 jam hanya untuk membangun fitur (swap), lalu container restart sekitar 00:00 WIB.
  Run #1 baru menyelesaikan fold 0 dari tiga anggota pertama; tidak ada artefak final yang terpakai.
- Tindakan: RAM dinaikkan ke 32 GB (user). Run #1 diluncurkan ulang dengan --clean (00:36 WIB, PID 4476).
  E0260 sampai E0263 diluncurkan ulang lewat logs/poisson_ab.sh (smoke lalu full) dengan pengaman MemAvailable < 3 GB.
  E0210 (t5 cross_entropy) tercatat failed: dihentikan saat queue di-STOP, fold 0 0,07533, tidak diulang karena t5 sudah di floor binomial.
- Catatan: nvidia-smi gagal setelah restart (NVML 595 vs modul kernel 580), tetapi torch dan XGBoost CUDA berfungsi.
  Kesetaraan hasil GPU dengan array dev akan terlihat dari sha256 run #1 vs sub_v16dev.

### [2026-10-17 01:00 WIB] E0264-E0265 | Fase 3 | Poisson pada XGB (anggota tunggal terkuat)
- Hipotesis: objective Poisson (count:poisson, link log) pada laju per exposure mengalahkan MSE rasio E0190 (gas 0,25291, EUI 0,66444).
  Dasar: screening LGB (E0200-E0208) memberi Poisson gain terbesar (gas -0,0041, EUI -0,0049 vs identity). Noise label = rata-rata rumah tangga, varians sebanding mean.
- Setup: parameter XGB E0190 (lr 0,05, d8), FS_legacy + agg + TE, 5-fold primary.
  - E0264: gas per meter, bobot M (setara Poisson gas dengan offset log M).
  - E0265: EUI (gas per luas), bobot A (setara Poisson gas dengan offset log A).
  - min_child_weight disetarakan sekitar 50 baris: hessian Poisson = w x mu (mu sekitar 1e4 per meter, 135 per m2), jadi mcw 1e6 dan 1,35e4.
- Jalan otomatis setelah rantai E0260-E0263 selesai (logs/xgb_poisson_ab.sh, pengaman RAM yang sama).
