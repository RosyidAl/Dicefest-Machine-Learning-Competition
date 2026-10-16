# Fase 0: setup lingkungan dan data (selesai 16 Okt 11:30 WIB)

## Lingkungan
- env/ENVIRONMENT.md, env/requirements.lock.txt.
- Smoke test OK untuk XGBoost CUDA, CatBoost GPU, PyTorch CUDA, LightGBM, dan FAISS GPU.

## Data
- data/raw: hardlink read-only.
- Verifikasi md5, shape, dan hitungan prefix: 201 fitur + 2 ID spasial.
- Cache parquet identik dengan CSV.

## Kode inti
- src/metric.py (metrik resmi verbatim), src/seed.py, src/folds.py, src/tracking.py.
- scripts/validate_submission.py, update_reports.py, queue_worker.py.
- tests: 8 lulus, termasuk tes kebocoran.

## Laporan dan pencatatan
- File fold permanen: primary k5/k10 (identik dengan split train_lib) dan secondary k5 (zone).
- Registry di-backfill: 168 model lama, 17 blend, 4 post-processing.
- Report diinisialisasi; git berjalan.
