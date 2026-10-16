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
