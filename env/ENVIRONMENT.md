# Lingkungan komputasi (dicatat 2026-10-16 11:26 WIB)

## Hardware
- Server kampus, container LXC di Proxmox (kernel 7.0.14-4-pve), OS Ubuntu 26.04 LTS.
- CPU: Intel Xeon Silver 4216 @ 2.10 GHz, 16 vCPU online untuk container ini (nproc = 16).
- RAM: 24 GiB, swap 2 GiB. Disk: 148 GB (sekitar 39 GB kosong saat dicatat).
- GPU: NVIDIA RTX A4000 16 GB, driver 595.84, CUDA driver 13.2. GPU ini dipakai bersama container lain di host yang sama:
  sekitar 11 GB VRAM sudah terpakai oleh proses di luar container (tidak terlihat dari sini), jadi sisa VRAM efektif sekitar 5 GB.

## Software
- Python 3.12.13 di `.venv` (dikelola uv 0.x). Python sistem 3.14 tidak dipakai.
- Versi inti: numpy 2.5.4, pandas 3.0.6, pyarrow 26.0.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0,
  xgboost 3.4.1, catboost 1.2.10, torch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24), optuna 5.0.0, shap 0.53.0, plotly 7.1.0.
- Paket opsional terpasang: tabm 0.0.3, rtdl_num_embeddings 0.0.12, gpboost 1.7.4, umap-learn 0.5.12, pytabkit,
  faiss-gpu-cu12 1.14.1.
- Tidak dipasang: cuml-cu13 (dry-run akan menurunkan numpy 2.5.4, berisiko merusak env utama). AutoGluon belum dipasang
  (rencana venv terpisah `.venv-ag` bila dipakai).
- Kaggle CLI 2.2.4 terpasang tetapi belum ada kredensial; submit dilakukan manual oleh user (KAGGLE_AUTO_SUBMIT: false).
- pandoc: tidak tersedia.
- Daftar lengkap versi: `env/requirements.lock.txt`.

## Smoke test GPU (scripts/smoke_gpu.py)
Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.

  return func(**kwargs)
| Komponen | Status | Detail | Waktu |
|---|---|---|---|
| XGBoost CUDA | OK | xgboost 3.4.1, rmse 0.0961 | 2.0s |
| CatBoost GPU | OK | catboost 1.2.10, rmse 0.1380 | 1.1s |
| PyTorch CUDA | OK | torch 2.14.1+cu130, cuda 13.0, NVIDIA RTX A4000, sum -65952.4 | 1.6s |
| LightGBM CPU | OK | lightgbm 4.7.0, rmse 0.1159 | 0.3s |
| FAISS GPU | OK | faiss 1.14.1, gpus 1, knn ok (5, 3) | 0.3s |

## Data
- `data/raw/` berisi hardlink read-only ke file asli di `dataset/` (nama asli "train (1).csv" dst).
- md5: train fa14e9aa16ab8fa65a9a55e2e7188447, test 9511b3f7f8fcd912eb25cee8a109900d,
  sample_submission 0403eac2273404c8807422d1e3caa542.
- Shape: train 429.303 x 209, test 183.987 x 204, sample 183.987 x 6. Fitur: bld_dim 43, bld_era 15, hvac_dist 10,
  env_climate 6, socio_demog 127 = 201 fitur + zone_region_id + subzone_id (jadi angka 203 di Overview menghitung 2 ID spasial).
- `data/cache/*.parquet`: konversi lossless (pyarrow), identik kolom per kolom dengan CSV (env/data_check.json).
- Kolom `id` berupa string "ID_xxxxxx", tidak dipakai sebagai fitur.

## Proses lain di server
- tmux `pakpisi-queue` (/root/kaggle/scripts/queue_runner.sh) adalah job user untuk kompetisi lain. Jangan diganggu;
  perhitungkan pemakaian RAM/GPU-nya sebelum meluncurkan job.
