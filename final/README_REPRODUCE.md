# Reproduksi submisi final DICE FEST 2026

## Isi folder
| Path | Isi |
|---|---|
| `DICEFEST2026_<TEAM>.ipynb` | notebook utama mandiri: semua kode sumber tertanam (`%%writefile`), melatih semua model dan menulis submisi + sha256 |
| `inference_from_artifacts.ipynb` | notebook cadangan: membuat ulang submisi yang sama dari model tersimpan di `artifacts/` tanpa training |
| `run_all.py` | satu pintu baris perintah untuk pipeline yang sama (`python final/run_all.py --clean`) |
| `src/` | modul beku: data (`common.py`), target encoding (`te.py`), GBDT (`train_lib.py`), MLP (`nn.py`), blend NNLS (`blend.py`), adopsi (`adopt_v2.py`), koreksi residual (`postprocess.py`), duplikat (`dup_fix.py`), orkestrasi (`pipeline.py`), inference (`inference.py`) |
| `configs/members.yaml` | 50 model anggota blend: back-end, formulasi target, bobot, parameter, jumlah fold, fitur |
| `configs/resid_knn.json` | bobot koreksi residual (beku, hasil fit di holdout internal 70/30 train) |
| `artifacts/` | model per fold (XGBoost .ubj, LightGBM .txt, MLP .pt), prediksi OOF/test per anggota, `SHA256SUMS` |
| `requirements.txt` | versi paket yang dipin |
| `make_notebooks.py` | membangun kedua notebook dari `src/` dan `configs/` |

## Langkah
1. Siapkan Python 3.12 dan paket di `requirements.txt` (PyTorch build CUDA 13.0).
2. Letakkan `train.csv`, `test.csv`, `sample_submission.csv` di satu folder, lalu set `DICEFEST_DATA_DIR` ke folder itu (default `data/raw`).
3. Jalankan salah satu:
   - Notebook utama, headless:
     `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 DICEFEST2026_<TEAM>.ipynb --output executed.ipynb`
   - Baris perintah: `python final/run_all.py --work final/work/run1 --out final/output/submission_run1.csv --clean`
   - Verifikasi cepat dari artefak: jalankan `inference_from_artifacts.ipynb`. Tidak ada training kecuali model adopsi kecil (sekitar 2 menit).
4. Bandingkan sha256 output dengan nilai di bawah.

## Determinisme
- Seed dikunci di semua tahap:
  - KFold(shuffle=True, random_state=42);
  - XGBoost seed 42 (CUDA hist, deterministik);
  - LightGBM seed 42 dengan deterministic=True, force_row_wise=True, num_threads=12;
  - MLP: torch.manual_seed per seed dan fold, cudnn.deterministic, use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8;
  - PCA random_state=0.
- Tidak ada time limit. Semua early stopping berbasis iterasi.
- CatBoost GPU sengaja tidak dipakai karena tidak deterministik (DECISIONS D004).
- Hasil GPU bisa berbeda tipis antar jenis GPU atau versi driver. Hardware referensi tercatat di bawah.
  Bila GPU berbeda, notebook inference dari artefak memberi hasil identik karena hanya memakai prediksi dari model tersimpan.

## Hardware dan waktu jalan referensi
- NVIDIA RTX A4000 16 GB (driver 595.84, CUDA 13.0), Intel Xeon Silver 4216 16 vCPU, RAM 24 GB, Ubuntu 26.04.
- Waktu jalan per tahap: diisi setelah run #1 (lihat `reports/phases/PHASE_11.md`).

## sha256 yang diharapkan
Diisi setelah audit (run #1, run #2, notebook headless).
