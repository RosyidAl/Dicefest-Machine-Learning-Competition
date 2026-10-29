# MASTER PROMPT: DICE FEST 2026 Machine Learning
## The Urban Energy Transition Trilemma (Multi-Output Regression)

Dokumen ini adalah instruksi kerja lengkap untuk Claude Code. Baca seluruhnya satu kali di awal. Setelah itu cukup baca ulang bagian fase yang sedang aktif. Aturan ringkas yang selalu berlaku ada di `CLAUDE.md`.

---

## 0. Konfigurasi

Boleh diubah user sebelum mulai. Jika ada konflik dengan bagian lain, blok ini yang berlaku.

```yaml
DATA_DIR: ./data/raw                  # train.csv, test.csv, sample_submission.csv
KAGGLE_COMPETITION: dicefest-2026-machine-learning
KAGGLE_AUTO_SUBMIT: false             # true = agen boleh submit sendiri via Kaggle CLI
MAX_AGENT_SUBMITS_PER_DAY: 3          # sisanya untuk user (limit Kaggle 5/hari)
GLOBAL_SEED: 42
SEED_BAG: [42, 2026, 7, 1234, 31337]
N_FOLDS: 5
TIMEZONE: Asia/Jakarta
GATE_FREEZE_FINAL: "2026-10-22 12:00 WIB"
GATE_PAKET_FINAL: "2026-10-23 12:00 WIB"
DEADLINE: "2026-10-23 23:59 WIB"
TEAM_NAME: "ISI_NAMA_TIM"
```

---

## 1. Peran, tujuan, cara berpikir

Kamu adalah lead ML engineer setingkat Kaggle Grandmaster yang bekerja otonom di server dengan GPU RTX A4000 16 GB dan RAM 24 GB. Tujuanmu satu: skor Column-Wise NRMSE serendah mungkin di Private Leaderboard, dengan pipeline final yang reproducible bit-per-bit, bebas kebocoran, dan terdokumentasi sehingga lolos audit juri dan bisa dipertahankan di presentasi final.

Waktu komputasi tidak dibatasi, tetapi deadline tetap. Jalankan sebanyak mungkin eksperimen yang bernilai, jangan biarkan GPU dan CPU menganggur, dan patuhi gerbang jadwal di bagian 17.

Urutan sumber gain (jangan lompat ke tuning sebelum nomor 1 sampai 3 matang):
1. Validasi yang meniru cara test dibuat. Salah di sini berarti semua keputusan berikutnya salah arah.
2. Pemahaman struktur data dan target: relasi fisik antar target, level spasial fitur, komposisi.
3. Transformasi target dan feature engineering.
4. Model yang kuat dan beragam.
5. Hyperparameter tuning.
6. Multi-seed, ensemble, stacking.

Sikap ilmiah: setiap ide adalah hipotesis. Tulis hipotesisnya, uji dengan fold yang sama, terima atau tolak berdasarkan angka, lalu catat. Ide yang gagal juga dicatat karena berguna untuk writeup dan mencegah pengulangan.

"Setiap pendekatan dicoba" diwujudkan lewat `reports/COVERAGE.md`: daftar semua pendekatan di dokumen ini (T0 sampai T10, FB01 sampai FB12, semua keluarga model, semua metode ensemble, teknik lanjutan) dengan status DONE (exp_id + hasil), SKIPPED (alasan berbasis bukti), RUNNING, atau QUEUED. Di akhir proyek tidak boleh ada item QUEUED atau RUNNING.

---

## 2. Fakta kompetisi

### 2.1 Data
- `train.csv`: 429.303 baris x 209 kolom = `id` + 201 fitur numerik + `zone_region_id` + `subzone_id` + 5 target.
- `test.csv`: 183.987 baris x 204 kolom (tanpa target).
- `sample_submission.csv`: 183.987 baris x 6 kolom.
- Halaman Overview menyebut 203 fitur, halaman Data menyebut 201. Hitungan prefix 43 + 15 + 10 + 6 + 127 = 201, jadi angka 203 kemungkinan ikut menghitung 2 ID spasial. Verifikasi di Fase 0.
- Prefix fitur:
  - `bld_dim_*` (43): dimensi dan geometri bangunan (luas lantai GIA, selubung GEA, tinggi, jumlah lantai, volume, rasio bentuk, tipologi tapak).
  - `bld_era_*` (15): sebaran era konstruksi, pra-1919 sampai pasca-1999, penentu kualitas insulasi.
  - `hvac_dist_*` (10): bauran sistem pemanas (gas pipa, minyak, listrik, biomassa, komunal).
  - `env_climate_*` (6): beban termal iklim mikro (HDD/CDD).
  - `socio_demog_*` (127): okupansi, kepadatan, aktivitas rumah tangga, tenure (milik sendiri, sewa sosial, sewa swasta).
- Setiap baris adalah agregat satu kawasan permukiman (sub-zona). `zone_region_id`: 330 klaster wilayah. `subzone_id`: sub-zona mikro bertingkat.
- Nama fitur disamarkan. Semantik kolom harus ditemukan lewat analisis (Fase 1).

### 2.2 Target (urutan wajib di submisi)

| # | Kolom | Satuan dan skala | Sigma beku |
|---|---|---|---|
| t1 | `target_macro_gas_kwh` | kWh/tahun, sekitar 1e4 sampai 1e6 | 151750.04 |
| t2 | `target_macro_elec_kwh` | kWh/tahun, sekitar 1e4 sampai 1e5 | 41725.37 |
| t3 | `target_eui_gas_kwh_per_sqm` | kWh/m2, sekitar 1e1 sampai 1e2 | 40.65 |
| t4 | `target_clean_building_ratio` | persen 0 sampai 100 | 12.49 |
| t5 | `target_electric_heating_adoption` | persen 0 sampai 100 | 7.18 |

### 2.3 Metrik

Skor = (1/5) x jumlah RMSE_k / sigma_k untuk k = t1 sampai t5. Lebih kecil lebih baik. Pakai implementasi resmi ini verbatim di `src/metric.py`:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

TARGET_COLS = [
    'target_macro_gas_kwh',
    'target_macro_elec_kwh',
    'target_eui_gas_kwh_per_sqm',
    'target_clean_building_ratio',
    'target_electric_heating_adoption'
]

OFFICIAL_SIGMAS = {
    'target_macro_gas_kwh': 151750.04,
    'target_macro_elec_kwh': 41725.37,
    'target_eui_gas_kwh_per_sqm': 40.65,
    'target_clean_building_ratio': 12.49,
    'target_electric_heating_adoption': 7.18
}

def compute_column_wise_nrmse(y_true, y_pred, sigmas=OFFICIAL_SIGMAS):
    y_true_vals = y_true[TARGET_COLS].values if isinstance(y_true, pd.DataFrame) else np.array(y_true)
    y_pred_vals = y_pred[TARGET_COLS].values if isinstance(y_pred, pd.DataFrame) else np.array(y_pred)
    nrmse_list = []
    for idx, col in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_true_vals[:, idx], y_pred_vals[:, idx]))
        nrmse_list.append(rmse / sigmas[col])
    return float(np.mean(nrmse_list))
```

Konsekuensi:
- Metrik terpisah per kolom. Setiap target boleh punya feature set, transform, model, dan ensemble sendiri. Meminimalkan MSE per target sama dengan meminimalkan metrik.
- Prediksi konstan (rata-rata) memberi NRMSE sekitar 1.0 per target. Ini sanity check.
- RMSE dihitung di skala asli. Untuk target makro, baris bernilai besar mendominasi error. Transform log boleh dipakai saat training, tetapi evaluasi SELALU di skala asli.
- Kontribusi tiap target ke skor akhir = NRMSE_k / 5. Prioritaskan target dengan NRMSE terbesar.

### 2.4 Leaderboard dan submisi
- Public LB = 30% test, Private LB = 70% test. Finalis ditentukan 95% dari skor Private LB.
- 5 submisi per hari (reset 00:00 UTC = 07:00 WIB). Maksimal 2 submisi final dipilih.
- Header persis: `id,target_macro_gas_kwh,target_macro_elec_kwh,target_eui_gas_kwh_per_sqm,target_clean_building_ratio,target_electric_heating_adoption`. 183.987 baris. `id` dan urutannya sama dengan `sample_submission.csv`.

### 2.5 Aturan
- Hanya data resmi. Dilarang data eksternal, scraping sensus, data geografis tambahan.
- Dilarang memakai API LLM/foundation model tertutup atau berbayar (GPT-4, Claude, Gemini API) atau API tabular cloud proprietary untuk menghasilkan nilai prediksi. Dilarang hand-labeling. Claude Code boleh menulis kode, tetapi setiap nilai prediksi wajib keluar dari kode open-source yang dijalankan lokal.
- Library open-source bebas dipakai (scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch, TensorFlow, SciPy, Optuna, FLAML, Auto-Sklearn, H2O, dan sejenisnya).
- Seed wajib dikunci di semua library dan split. Output harus identik saat diaudit juri.
- Wajib menyerahkan notebook `.ipynb` yang menghasilkan file submisi pemenang secara deterministik, plus writeup PDF, lewat https://dicefest.web.id paling lambat 23 Oktober 2026 23:59 WIB.
- Timeline lanjutan: 24 Okt audit kode, 25 Okt pengumuman Top 10, 30 Okt presentasi final via Zoom (10 menit paparan + 10 menit tanya jawab), 31 Okt pengumuman juara.

### 2.6 Hipotesis awal yang WAJIB diverifikasi di Fase 1
- **H-SPLIT.** 429.303 / 613.290 = 0,700000 tepat, dan 183.987 = 0,3 x 613.290 tepat. Ini pola khas `train_test_split(test_size=0.3)` per baris. Jika benar, `zone_region_id` dan `subzone_id` di test hampir semuanya juga muncul di train. Akibatnya: informasi tingkat grup (target encoding, agregat grup, embedding ID) sah dan sangat berharga, dan CV acak per baris lebih meniru test daripada GroupKFold.
- **Saran penyelenggara.** Spatial GroupKFold per `zone_region_id` tetap dijalankan sebagai CV sekunder dan dilaporkan. Namun CV utama untuk seleksi model harus meniru mekanisme split yang terbukti dari data. Jika data membuktikan split berbasis grup, CV utama ikut grup.
- **H-FISIKA.** t1 kemungkinan sekitar t3 x total luas lantai (salah satu kolom `bld_dim_*`). Target makro berskala dengan ukuran kawasan (luas, jumlah hunian). t5 kemungkinan sangat terkait porsi pemanas listrik di `hvac_dist_*`. t4 kemungkinan terkait era konstruksi dan tenure.

---

## 3. Arsitektur proyek

### 3.1 Struktur folder

```text
.
├── CLAUDE.md
├── MASTER_PROMPT.md
├── data/
│   ├── raw/          CSV asli, read-only
│   ├── cache/        parquet hasil konversi
│   └── folds/        file fold permanen (id, fold)
├── features/         feature store: <FS_ID>/manifest.json + parquet per blok (per fold untuk blok ber-target)
├── configs/          YAML per eksperimen, best_params/, final/
├── src/
│   ├── metric.py folds.py data.py targets.py seed.py tracking.py train_cv.py submit.py
│   ├── features/     satu modul per blok fitur
│   ├── models/       lgb.py xgb.py cat.py nn/ knn.py linear.py gpboost.py ...
│   └── ensemble/     blend.py hillclimb.py stack.py postproc.py
├── scripts/          update_reports.py queue_worker.py validate_submission.py check_repro.py
├── tests/
├── experiments/
│   ├── registry.csv
│   └── E0001_<slug>/ config.yaml run.log metrics.json oof.parquet test_pred.parquet
│                     feature_importance.csv env.json README.md plots/ models/ (opsional)
├── queue/{gpu,cpu}/{pending,running,done,failed}/
├── optuna/           storage SQLite per study
├── submissions/      submission_log.csv, BEST_CURRENT.csv, file kandidat
├── reports/          STATUS JOURNAL LEADERBOARD DECISIONS BACKLOG COVERAGE NEEDS_USER (.md)
│                     dashboard.html, eda/, phases/, figures/
├── final/            pipeline final, notebook, artefak, README_REPRODUCE.md
├── env/              requirements.lock.txt, ENVIRONMENT.md
└── logs/
```

### 3.2 Kontrak modul inti
- `src/metric.py`: fungsi resmi di atas, plus `nrmse_per_target(y_true, y_pred, targets)` yang mengembalikan dict per target dan bisa menilai subset target. Unit test: sama dengan implementasi numpy manual; prediksi rata-rata memberi sekitar 1.0.
- `src/folds.py`: membuat dan memuat file fold; validasi (tiap id tepat di satu fold, ukuran seimbang, pada skema grup tidak ada grup yang terbelah).
- `src/data.py`: memuat cache parquet; fitur float32, target float64; daftar fitur per prefix yang di-sort; `load_feature_set(fs_id)`.
- `src/targets.py`: transform target dengan `forward(y, ctx) -> (y_t, weight)` dan `inverse(pred_t, ctx) -> pred`; `ctx` membawa kolom exposure bila perlu. Jenis: identity, log1p, sqrt, yeo-johnson (fit di train fold), ratio-to-exposure, logit-percent, proportion (y/100).
- `src/models/*`: antarmuka seragam `fit(X_tr, y_tr, X_va, y_va, w_tr=None)`, `predict(X)`, `best_iteration`, `feature_importance()`. Semua parameter dan seed berasal dari config.
- `src/features/*`: setiap blok punya `uses_target: bool` dan `level: "transductive" | "fold"`. Blok transductive (tanpa target) dihitung sekali di train+test. Blok fold (memakai target) dihitung di dalam loop fold dengan cross-fitting: nilai untuk baris training berasal dari inner K-fold di train-part, nilai untuk validasi dan test berasal dari seluruh train-part fold tersebut.
- `src/train_cv.py`: `python -m src.train_cv --config configs/E0042.yaml [--smoke]`. Untuk setiap target dan fold: bangun fitur fold-aware, transform target, fit, prediksi validasi dan test, inverse transform, clip. Hasil: OOF lengkap, prediksi test = rata-rata model fold, metrik per fold per target. Artefak dan baris registry ditulis otomatis. Error ditangkap: status failed + traceback di run.log.
- `src/tracking.py`: exp_id berurutan (E0001, ...), git commit, hash config, hash data, hash daftar fitur, versi library, device, waktu mulai dan selesai (WIB), peak RAM (psutil), peak VRAM.
- `src/submit.py` dan `scripts/validate_submission.py`: membangun CSV dari prediksi test, mengurutkan sesuai sample_submission, memvalidasi (Lampiran G), menulis sha256.
- `scripts/update_reports.py`: membangun ulang `reports/LEADERBOARD.md` dan `reports/dashboard.html` (Plotly, satu file statis) dari registry dan submission_log.
- `scripts/queue_worker.py`: lihat 3.4.
- `scripts/check_repro.py`: menjalankan config yang sama dua kali di proses terpisah, membandingkan sha256 dan max |selisih| OOF serta prediksi test.

### 3.3 Sistem pencatatan

| File | Isi | Kapan diperbarui |
|---|---|---|
| `experiments/registry.csv` | satu baris per eksperimen (kolom di Lampiran A) | otomatis oleh runner |
| `experiments/<id>/README.md` | hipotesis, setup, hasil per target vs terbaik, kesimpulan | otomatis + kesimpulan |
| `reports/JOURNAL.md` | log naratif kronologis (Lampiran C) | sebelum dan sesudah tiap eksperimen atau batch |
| `reports/LEADERBOARD.md` | terbaik global, terbaik per target, top-10 per keluarga model, 20 eksperimen terakhir | otomatis |
| `reports/STATUS.md` | posisi terkini untuk resume (Lampiran E) | setiap selesai satu langkah |
| `reports/DECISIONS.md` | keputusan penting format ADR (Lampiran D) | saat keputusan diambil |
| `reports/BACKLOG.md` | antrean ide: prioritas P0/P1/P2, estimasi gain, biaya, status | terus-menerus |
| `reports/COVERAGE.md` | checklist semua pendekatan dan statusnya | setiap ada perubahan status |
| `reports/NEEDS_USER.md` | hal yang butuh tindakan user | saat muncul |
| `reports/phases/PHASE_<n>.md` | ringkasan fase: yang dicoba, yang berhasil, angka | akhir fase |
| `reports/phases/GATE_<n>.md` | ringkasan gerbang jadwal | di setiap gerbang |
| `reports/dashboard.html` | grafik skor terbaik vs waktu, per target, per keluarga, CV vs LB | otomatis |
| `submissions/submission_log.csv` | file, sha256, sumber, CV per target, public LB, catatan | setiap kandidat dibuat atau di-submit |

Prinsip: setiap angka di laporan bisa ditelusuri ke exp_id, dan setiap exp_id bisa ditelusuri ke config, commit git, file fold, dan feature set.

### 3.4 Eksekusi job
- Queue worker dua jalur: `queue/gpu` dan `queue/cpu`. Item = file YAML config. Worker mengambil item tertua di `pending`, memindahkannya ke `running`, menjalankan `python -m src.train_cv --config <item>`, lalu memindahkan ke `done` atau `failed`. Satu item per jalur pada satu waktu.
- Menjalankan worker: `nohup python scripts/queue_worker.py --lane gpu > logs/worker_gpu.log 2>&1 &` (dan `--lane cpu`). Simpan PID di `logs/workers.pid`.
- Sebelum analisis panjang atau malam hari, isi antrean dengan beberapa eksperimen bernilai agar GPU tidak menganggur.
- Pemantauan: `tail -n 30 <log>`, `ls queue/*/running`, `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`.
- Worker mati: deteksi lewat PID, jalankan ulang, kembalikan item `running` ke `pending` dengan catatan di JOURNAL.
- Menunggu: kerjakan persiapan lain dulu. Jika memang harus menunggu, `sleep` paling lama 9 menit per perintah (atur timeout perintah), lalu cek lagi.

### 3.5 Manajemen resource
- Cek di awal: `nproc`, `lscpu`, `free -g`, `nvidia-smi`, `df -h`. Simpan di `env/ENVIRONMENT.md`.
- Thread: total thread semua job CPU yang paralel tidak melebihi jumlah core fisik. Set `num_threads`, `n_jobs`, `thread_count` secara eksplisit.
- RAM: fitur float32, hindari salinan DataFrame, `del` + `gc.collect()`. Maksimal 2 sampai 3 proses yang memuat data penuh bersamaan. Cek `free -g` sebelum meluncurkan job.
- VRAM: satu job GPU berat pada satu waktu. Job GPU kecil (KNN, PCA) boleh berbarengan hanya jika memori terpakai di bawah 60%. Setelah job PyTorch: `torch.cuda.empty_cache()`.
- Disk: cek `df -h` tiap hari. Simpan binary model hanya untuk kandidat final.

### 3.6 Standar reproducibility

Dipanggil di awal setiap proses, sebelum CUDA dipakai:

```python
import os, random
import numpy as np

def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # harus sebelum CUDA diinisialisasi
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
```

- Urutan kolom dan iterasi apa pun selalu dari list yang di-sort, jangan dari `set()` atau urutan yang bergantung pada hashing string.
- LightGBM: `seed`, `deterministic=True`, `force_row_wise=True` (atau `force_col_wise=True`), `num_threads` tetap.
- XGBoost: `random_state`, `device="cuda"`, `tree_method="hist"`. Determinisme GPU dibuktikan empiris dengan `check_repro.py`.
- CatBoost: `random_seed`, `thread_count` tetap. Determinisme GPU dibuktikan empiris.
- scikit-learn: `random_state` di semua splitter, PCA/SVD acak, KMeans (`n_init` eksplisit), `TargetEncoder`, `QuantileTransformer`.
- PyTorch: DataLoader dengan `generator=torch.Generator().manual_seed(seed)` dan `worker_init_fn`, atau `num_workers=0`. AMP hanya jika lolos `check_repro.py`.
- Optuna: `TPESampler(seed=...)`. Proses tuning tidak wajib reproducible, tetapi parameter terbaik dibekukan ke YAML dan model final dilatih ulang dari YAML.
- Dilarang di pipeline final: `time_limit`, `max_runtime_secs`, early stopping berbasis waktu.
- Uji determinisme per keluarga model sejak Fase 2, bukan di akhir.

---

## 4. Fase 0: Setup lingkungan dan data

1. Jalankan `TZ=Asia/Jakarta date`, `nvidia-smi`, `nproc`, `lscpu | head -20`, `free -h`, `df -h .`, `python3 --version`, `which uv conda pip`. Tulis hasilnya di `env/ENVIRONMENT.md`.
2. Buat struktur folder (3.1). `git init`. `.gitignore`: `data/`, `features/`, `experiments/*/models/`, `*.parquet`, `*.npy`, `*.pkl`, `optuna/*.db`, `queue/`, `logs/`, `final/artifacts/`, `.venv*`.
3. Environment utama (venv; pakai `uv` bila tersedia): numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, xgboost, catboost, torch (build CUDA yang cocok dengan driver), optuna, shap, matplotlib, plotly, pyyaml, psutil, tqdm, pytest, jupyter, nbconvert, ipykernel, kaggle.
4. Paket opsional, install terpisah satu per satu; jika gagal, catat dan lanjut tanpa merusak env utama: cuml (sesuai versi CUDA), faiss-gpu, pytabkit, tabm, rtdl_num_embeddings, gpboost, umap-learn.
5. AutoGluon (bila dipakai) di venv terpisah `.venv-ag`, karena biasanya mem-pin versi lightgbm, xgboost, catboost, torch yang berbeda.
6. `pip freeze > env/requirements.lock.txt` (juga untuk `.venv-ag`).
7. Smoke test GPU: fit kecil XGBoost `device="cuda"`, CatBoost `task_type="GPU"`, perkalian matriks PyTorch di CUDA, LightGBM CPU. Catat hasil dan versi.
8. Data: jika `DATA_DIR` kosong dan Kaggle CLI terkonfigurasi (`~/.kaggle/kaggle.json`), unduh dengan `kaggle competitions download -c dicefest-2026-machine-learning -p data/raw` lalu unzip. Jika tidak bisa, tulis di `reports/NEEDS_USER.md` dan berhenti, karena tanpa data tidak ada yang bisa dikerjakan.
9. Verifikasi shape (429.303 x 209, 183.987 x 204, 183.987 x 6), nama kolom, hitungan per prefix, md5 file. Konversi ke `data/cache/*.parquet` secara lossless (float64 apa adanya). Konversi ke float32 dilakukan saat load untuk training.
10. Implementasi `src/metric.py`, `src/seed.py`, `src/tracking.py`, `scripts/validate_submission.py`, beserta unit test (pytest). Format `sample_submission.csv` harus lolos validator (cek rentang nilai boleh dilewati untuk file contoh).
11. Inisialisasi file laporan (STATUS, JOURNAL, DECISIONS, BACKLOG, COVERAGE, NEEDS_USER, LEADERBOARD) dan `experiments/registry.csv` dengan header. COVERAGE.md langsung diisi seluruh daftar pendekatan dari dokumen ini dengan status QUEUED.

Kriteria selesai: env berfungsi dan tercatat, GPU terverifikasi, data ter-cache dan terverifikasi, metrik teruji, git berjalan, STATUS.md terisi.

---

## 5. Fase 1: EDA dan forensik data

Keluaran: `reports/eda/EDA_REPORT.md`, figur di `reports/eda/figures/`, angka mentah di `reports/eda/findings.json`. Setiap temuan ditutup dengan "Implikasi untuk pemodelan" yang langsung menjadi item BACKLOG.

### 5.A Integritas
- dtype, NaN, inf, nilai sentinel (misal -999, atau nol yang berlebihan), kolom konstan, kolom duplikat persis, pasangan kolom dengan |r| > 0,999.
- Baris duplikat (fitur saja) di train, di test, dan lintas train-test. Untuk duplikat di train, seberapa mirip targetnya (estimasi noise label).
- Proporsi nilai integer per kolom (kandidat count atau kode kategori).

### 5.B Forensik split (paling penting)
- Rentang dan urutan `id` di train dan test, celah, monotonisitas.
- Korelasi Spearman `id` dengan setiap target dan dengan `zone_region_id`; plot rolling mean target terhadap `id`. Tujuannya memastikan `id` tidak membawa informasi. Jika ternyata membawa informasi, catat di NEEDS_USER, dan `id` tetap tidak dipakai sebagai fitur.
- `zone_region_id`: jumlah unik di train, test, dan irisannya; sebaran jumlah baris per region; proporsi baris test yang region-nya ada di train; untuk tiap region, rasio n_test / (n_train + n_test). Split acak per baris akan memberi rasio yang menyebar binomial di sekitar 0,3.
- `subzone_id`: analisis yang sama. Cek hierarki (apakah tiap subzone hanya ada di satu region), apakah kode subzone mengandung kode region, dan jumlah baris per subzone (banyak baris per subzone, atau hampir unik per baris).
- Adversarial validation: LightGBM membedakan train vs test (5-fold stratified), dengan dan tanpa ID spasial. Laporkan AUC dan fitur yang paling membedakan. AUC sekitar 0,5 berarti distribusinya sama.
- Simpulkan mekanisme split: acak per baris, grup per subzone, grup per region, atau lainnya. Tulis bukti angkanya di `reports/DECISIONS.md` sebagai D001.

### 5.C Target
- Statistik deskriptif, kuantil (0,1%, 1%, 50%, 99%, 99,9%), skewness, jumlah nol, nilai minimum, pelanggaran rentang untuk target persen.
- Bandingkan std train (ddof 0 dan 1) dengan sigma beku. Catat jika selisihnya besar.
- Histogram skala asli dan log; korelasi Pearson dan Spearman antar target.
- Kontribusi ekor: berapa persen total kuadrat deviasi berasal dari 1% baris terbesar. Ini menunjukkan seberapa besar RMSE akan didominasi nilai ekstrem.
- Dekomposisi variansi spasial: prediktor "rata-rata target per region" dan "per subzone" yang dihitung OOF (5-fold acak), lalu NRMSE-nya. Ini mengukur besarnya sinyal di level grup.

### 5.D Fitur
- Statistik per prefix: skala, rentang, nilai negatif, sebaran.
- Deteksi komposisi: jumlah per baris untuk setiap prefix dan untuk blok kolom bersebelahan. Cari blok yang jumlahnya konstan (misal 1 atau 100) atau sama dengan kolom lain. Hasilnya: daftar blok "share" dan kolom "total" (kandidat jumlah hunian atau luas total).
- Level spasial fitur: rasio variansi dalam-region terhadap variansi total untuk setiap fitur. Fitur yang konstan di dalam region (misal iklim) adalah atribut region.
- Klaster korelasi (Spearman) untuk mendeteksi kelompok fitur redundan.
- Relasi univariat dengan setiap target: Spearman, mutual information (subsample), importance LightGBM cepat per target.

### 5.E Penemuan struktur fisik
- Untuk setiap kolom `bld_dim_*` yang positif: R2 dari log(t1) ~ log(t3) + log(f), dan stabilitas rasio t1 / (t3 x f). Kolom dengan rasio paling stabil adalah kandidat total luas lantai (exposure untuk target gas).
- Kandidat exposure untuk t2: kolom count/total dengan rasio t2 / f paling stabil.
- Seberapa jauh t5 dijelaskan `hvac_dist_*` (regresi linear dan LightGBM kecil), dan t4 oleh `bld_era_*` + `hvac_dist_*` + tenure.
- Apakah urutan kolom `bld_era_*` kronologis (cek hubungan monoton dengan t3).
- Laporkan setiap relasi yang hampir deterministik (R2 > 0,95) beserta rumusnya.

### 5.F Ringkasan
- 10 sampai 20 temuan terpenting, masing-masing dengan angka dan implikasinya. Isi BACKLOG dengan prioritas.

Kriteria selesai: EDA_REPORT lengkap, D001 (mekanisme split) tertulis dengan bukti, BACKLOG terisi.

---

## 6. Fase 2: Desain validasi dan baseline

### 6.1 Memilih CV utama (CV_PRIMARY)
Tiru mekanisme split test yang terbukti di Fase 1:
- Acak per baris: StratifiedKFold(5, shuffle=True, random_state=42) pada 20 bin kuantil t1, agar baris ekor tersebar merata antar fold.
- Grup per subzone: GroupKFold per `subzone_id`.
- Grup per region: GroupKFold per `zone_region_id`.
- Campuran atau tidak jelas: splitter yang meniru proporsi "grup terlihat vs tidak terlihat" yang teramati. Dokumentasikan.
- Untuk skema grup, alokasi grup ke fold dibuat acak ber-seed sambil menyeimbangkan jumlah baris per fold.

CV_SECONDARY: GroupKFold per `zone_region_id` (saran penyelenggara). Dipakai untuk melaporkan ketahanan terhadap wilayah yang tidak terlihat dan sebagai lindung nilai saat memilih final. Jika CV_PRIMARY sudah berbasis region, CV_SECONDARY = KFold acak.

Simpan `data/folds/folds_primary_k5_s42.parquet` dan `data/folds/folds_secondary_k5_s42.parquet` (kolom `id`, `fold`). Uji integritasnya di `tests/test_folds.py`. File ini permanen.

### 6.2 Kriteria menerima perbaikan
- Untuk target k, perubahan diterima jika NRMSE_k OOF turun DAN (membaik di minimal 4 dari 5 fold ATAU penurunannya lebih dari 2 x standard error selisih per fold).
- Perubahan yang menambah kompleksitas besar harus menurunkan NRMSE_k minimal 0,2% relatif. Selisih yang lebih kecil dianggap noise.
- Mode screening (3 fold) hanya untuk menyaring ide. Keputusan akhir selalu di 5 fold dengan kriteria ini.

### 6.3 Baseline (kelima target, di CV_PRIMARY dan CV_SECONDARY)
- B0: prediksi rata-rata fold training (sanity sekitar 1,0).
- B1: Ridge (StandardScaler, log1p untuk fitur non-negatif yang sangat miring).
- B2: LightGBM referensi (Lampiran H) tanpa ID spasial.
- B3: B2 + `zone_region_id` dan `subzone_id` (varian numerik dan varian kategori).
- B4: XGBoost GPU dengan parameter default yang wajar.
- B5: CatBoost GPU dengan `cat_features` = ID spasial.
- B6: MLP multi-output sederhana (input QuantileTransformer, target distandarkan).

Tabel hasil: target x model x skema CV. Hitung gap CV_PRIMARY vs CV_SECONDARY per target.

### 6.4 Uji determinisme per keluarga
Jalankan `scripts/check_repro.py` untuk LightGBM, XGBoost GPU, CatBoost GPU, dan MLP (subsample 50 ribu baris, 2 fold). Catat keluarga mana yang identik bit-per-bit di DECISIONS.md. Untuk yang tidak identik, cari pengaturan yang membuatnya identik, atau jadwalkan versi CPU untuk pipeline final.

### 6.5 Submisi pertama dan kalibrasi CV vs LB
- Baseline terbaik → validasi → `submissions/BEST_CURRENT.csv` → submit (atau catat di NEEDS_USER) → catat di submission_log.
- Uji pembeda skema CV: submit pasangan model yang sama dengan dan tanpa informasi grup (B2 vs B3). Bandingkan selisih public LB dengan selisih yang diprediksi CV_PRIMARY dan CV_SECONDARY. Skema yang selisihnya cocok dengan LB adalah skema yang meniru test. Catat sebagai D002.

Kriteria selesai: CV_PRIMARY ditetapkan dengan bukti, baseline lengkap tercatat, status determinisme per keluarga diketahui, submisi pertama yang valid sudah ada.

---

## 7. Fase 3: Rekayasa target (per target)

Model acuan: LightGBM referensi + fitur mentah (+ ID spasial bila terbukti membantu di Fase 2). Setelah pemenang ditemukan, konfirmasi ulang dengan CatBoost dan XGBoost, karena transform terbaik bisa berbeda per keluarga. Evaluasi selalu NRMSE di skala asli.

| Kode | Transform atau objective | Untuk | Catatan |
|---|---|---|---|
| T0 | identity, L2 | semua | baseline |
| T1 | log1p, inverse expm1 | t1, t2, t3 | uji koreksi bias: faktor smearing dan kalibrasi linear OOF (nested: parameter dipelajari dari fold lain) |
| T2 | sqrt | t1, t2, t3 | |
| T3 | Yeo-Johnson (lambda di-fit di train fold) | t1, t2, t3 | |
| T4 | rasio terhadap exposure: r = y / E, prediksi = r_hat x E | t1, t2 | E dari 5.E. Bobot sampel E^p dengan p = 0, 1, 2 (p = 2 membuat loss setara MSE skala asli). Normalisasi bobot ke rata-rata 1 |
| T5 | objective Tweedie (power 1,1 sampai 1,9), Gamma, Poisson | t1, t2, t3 | |
| T6 | Huber atau pseudo-Huber | semua | hanya jika outlier label terbukti noise |
| T7 | proporsi: y/100 dengan objective `cross_entropy` (LightGBM) atau `reg:logistic` (XGBoost), prediksi x 100 | t4, t5 | menjamin rentang 0 sampai 100 |
| T8 | logit((y + eps) / (100 + 2 eps)) | t4, t5 | uji beberapa eps |
| T9 | hurdle: P(y > 0) x E[y bila y > 0] | target dengan banyak nol | hanya jika 5.C menemukan zero-inflation |
| T10 | target turunan: t1 = t3_hat x E, dan t3 = t1_hat / E | t1, t3 | jika relasi 5.E kuat; bandingkan model langsung, turunan, dan blend keduanya |

Post-processing yang diuji di sini (nested): clip ke rentang valid; kalibrasi linear y = a + b x pred per target.

Keluaran: `reports/phases/PHASE_3.md` (tabel target x varian) dan D003 (transform terbaik per target, plus peringkat kedua untuk keragaman ensemble).

---

## 8. Fase 4: Feature engineering

### 8.1 Protokol ablation
- Model acuan tetap: LightGBM referensi, transform terbaik per target, CV_PRIMARY.
- Mode screening untuk banyak ide: learning_rate 0,1 dan 3 fold pertama. Ide yang lolos dikonfirmasi dengan 5 fold penuh dan learning_rate 0,05.
- Tambahkan satu blok setiap kali ke feature set terbaik saat ini. Catat delta per target. Feature set boleh berbeda per target.
- Lakukan juga ablation penghapusan per keluarga prefix, untuk memahami kontribusi tiap keluarga (bahan writeup).
- Setiap feature set punya ID (FS0001, ...) dengan manifest: blok, parameter, hash kode, jumlah fitur.

### 8.2 Blok fitur (urutan prioritas awal, boleh diubah oleh bukti Fase 1)
- **FB01 Pembersihan**: buang kolom konstan dan duplikat persis.
- **FB02 ID spasial**: numerik, kategori, frequency encoding (dihitung di train+test), jumlah subzone per region.
- **FB03 Komposisi**: share yang dinormalisasi; entropi dan indeks Herfindahl per keluarga (era, hvac, tenure, dan blok share lain yang ditemukan); kategori dominan; indeks era berbobot (jika urutan kronologis terbukti); share kumulatif era (misal pra-1945, pra-1980).
- **FB04 Fisika energi** (sesuaikan dengan semantik yang ditemukan): log luas, luas per hunian, rasio GEA/GIA (faktor bentuk), beban pemanasan = luas x HDD, beban gas = luas x HDD x share pemanas gas, proksi insulasi = share era tua x HDD, proksi kehilangan panas = GEA x HDD x share era tua, proksi beban listrik = jumlah hunian x share pemanas listrik x HDD. Jika semantik belum pasti, bangkitkan kandidat generik: perkalian top-k fitur penting dengan setiap kolom `env_climate_*`.
- **FB05 Agregat grup transductive** (tanpa target): untuk top-30 fitur penting, per region dan per subzone: mean, std, median, min, max; selisih dan rasio terhadap mean grup; rank persentil dalam grup; ukuran grup. Dihitung di train+test.
- **FB06 Target encoding OOF**: untuk setiap target (termasuk TE target lain sebagai fitur, yaitu cross-target), per region, per subzone, dan hierarkis (subzone di-shrink ke region, region ke global). Statistik: mean, median, std, count. Smoothing: auto, 10, 100. Untuk target makro, TE juga pada log-target dan pada rasio target/exposure. Implementasi: di dalam setiap fold, nilai untuk baris training dihitung dengan inner 5-fold di train-part, nilai untuk validasi dan test dihitung dari seluruh train-part fold itu. `sklearn.preprocessing.TargetEncoder` dengan `fit_transform` sudah memakai pola cross-fitting ini untuk mean; statistik lain diimplementasikan sendiri dengan pola yang sama.
- **FB07 KNN-target OOF** (GPU): ruang fitur terstandar (QuantileTransformer normal atau PCA whitening). k = 5, 20, 50, 200. Statistik target tetangga: mean, mean berbobot jarak, std, median; plus jarak rata-rata. Varian global dan varian dalam-region. Pola cross-fitting sama dengan FB06. Implementasi dengan cuML, FAISS, atau PyTorch (per chunk).
- **FB08 Reduksi dimensi transductive**: PCA per keluarga dan global (8 sampai 32 komponen), TruncatedSVD; opsional UMAP (cuML); opsional denoising autoencoder (swap noise) sebagai embedding.
- **FB09 Klaster**: KMeans k = 16, 64, 256 di train+test, menghasilkan ID klaster (kategori) dan jarak ke centroid terdekat; plus TE OOF dari ID klaster.
- **FB10 Pencarian interaksi**: operasi tambah, kurang, kali, bagi antar top-25 fitur, disaring dengan LightGBM cepat (subsample 100 ribu, 3 fold); simpan top-N. Opsional OpenFE pada subsample.
- **FB11 Transform untuk NN dan linear**: QuantileTransformer, log1p, rank-gauss.
- **FB12 Seleksi fitur**: null importance (permutasi target 20 sampai 50 kali), permutation importance di fold validasi, SHAP; buang fitur yang merugikan; buang fitur yang drift jika AUC adversarial tinggi. Seleksi dilakukan per target.

### 8.3 Tes kebocoran untuk blok yang memakai target
Acak label baris validasi di satu fold, lalu hitung ulang fitur fold itu. Nilai fitur baris validasi tidak boleh berubah sedikit pun. Wajib lulus sebelum blok dipakai (`tests/test_no_leakage.py`).

Keluaran: `reports/phases/PHASE_4.md` (tabel ablation per target) dan D004 (feature set final per target).

---

## 9. Fase 5: Model zoo

Semua model memakai CV_PRIMARY, feature set dan transform terbaik per target (variannya boleh dicoba), dan menyimpan OOF + prediksi test supaya bisa masuk ensemble.

### 9.1 Wajib
- **LightGBM (CPU)**: gbdt; varian `linear_tree=True` (bisa membantu target makro yang berskala linear dengan luas); varian `extra_trees=True`; varian GOSS (`data_sample_strategy="goss"`); ID spasial sebagai kategori vs sebagai TE.
- **XGBoost (GPU)**: `device="cuda"`, `tree_method="hist"`; `grow_policy` depthwise dan lossguide; `max_bin` 256 dan 1024; kategori native (`enable_categorical=True`) untuk ID spasial.
- **CatBoost (GPU)**: `task_type="GPU"`, `cat_features` = ID spasial (ordered target statistics bawaan CatBoost aman dari kebocoran di dalam training); depth 6 sampai 10; `grow_policy` SymmetricTree, Depthwise, Lossguide; `border_count` sampai 254.
- **MLP multi-output (PyTorch)** dengan embedding numerik periodic atau piecewise-linear (`rtdl_num_embeddings`) dan embedding ID spasial. Bandingkan loss MSE pada target terstandar dengan loss metrik langsung: rata-rata_k sqrt(MSE_batch_k) / sigma_k.
- **TabM** (paket `tabm` atau lewat `pytabkit`) dan **RealMLP** (`pytabkit`, misal `RealMLP_TD_Regressor`). `pytabkit` juga menyediakan default GBDT yang sudah di-tuning (LGBM_TD, XGB_TD, CatBoost_TD); jadikan pembanding.

### 9.2 Disarankan
- CatBoost `MultiRMSE` dan XGBoost `multi_strategy="multi_output_tree"` sebagai model multi-output untuk keragaman. Cek dukungan GPU di versi terpasang; jika tidak didukung, pakai CPU.
- **GPBoost**: boosting + random effects grup (region, dan subzone bersarang). Cocok dengan struktur hierarkis jika test berbagi grup dengan train.
- KNN regressor (cuML/FAISS), ExtraTrees (scikit-learn), Ridge di atas fitur kaya (spline, interaksi). Tujuannya keragaman ensemble, bukan menang sendirian.
- **AutoGluon** (venv terpisah) sebagai benchmark dan sumber OOF tambahan. Gunakan parameter `groups` berisi kolom fold kita supaya fold bagging-nya sama dengan fold kita (cek semantik parameter di versi terpasang). Preset `extreme` ditujukan untuk data kecil (puluhan ribu baris), jadi untuk 430 ribu baris pakai preset akurasi tertinggi yang cocok, misal `best` atau `best_v150` bila tersedia. Jangan pakai `time_limit` untuk run yang hasilnya dipakai. Jika hasilnya tidak identik saat diulang, AutoGluon hanya dipakai untuk eksplorasi, lalu konfigurasi terbaiknya diimplementasikan ulang secara eksplisit.

### 9.3 Eksploratif (jika waktu ada)
- FT-Transformer, ResNet-MLP.
- Foundation model tabular lokal (TabPFN, TabICL, Mitra, dan sejenisnya) pada subsample atau konteks KNN. Syarat: bobot dijalankan lokal (bukan API cloud) dan lisensinya open-source yang jelas. Jika lisensinya ambigu atau non-komersial, jangan dipakai di final tanpa persetujuan user (catat di NEEDS_USER).
- H2O AutoML (pakai `max_models`, bukan `max_runtime_secs`, supaya reproducible) atau FLAML sebagai benchmark.
- Distilasi: NN dilatih dengan target campuran label asli dan OOF ensemble.

### 9.4 Spesifikasi NN
- Input: fitur numerik melalui `QuantileTransformer(output_distribution="normal", random_state=42)` yang di-fit pada fitur train+test (unsupervised) atau train fold; ID spasial sebagai embedding (dimensi 8 sampai 32).
- Target: transform dari Fase 3, lalu standardisasi per target.
- Optimasi: AdamW, warmup + cosine atau OneCycle, batch 1024 sampai 8192, early stopping berdasarkan NRMSE resmi di fold validasi (setelah inverse transform), patience 10 sampai 20, maksimal 100 sampai 300 epoch.
- Regularisasi: dropout, weight decay; opsional mixup atau swap-noise.
- 3 sampai 5 seed per konfigurasi, lalu dirata-rata (variansi NN tinggi).
- Simpan bobot epoch terbaik per fold untuk kandidat final.

### 9.5 Analisis error (setelah zoo, dan diulang setelah ensemble)
- Per target: residual terbesar (baris mana, region mana, nilai fitur apa), residual per bin fitur penting, residual per region.
- Korelasi residual OOF antar model (bahan memilih komponen ensemble).
- Setiap pola yang ditemukan dirumuskan sebagai hipotesis baru di BACKLOG.

Keluaran: `reports/phases/PHASE_5.md` (tabel keluarga model x target, matriks korelasi residual, temuan analisis error).

---

## 10. Fase 6: Hyperparameter optimization

- Optuna: `TPESampler(seed=42, multivariate=True)`, `MedianPruner` dengan skor per fold sebagai intermediate value, storage `sqlite:///optuna/<study>.db`, `load_if_exists=True`.
- Satu study per (target, keluarga model, feature set). Prioritas: dua keluarga terbaik per target dari Fase 5, dimulai dari target dengan NRMSE terbesar (gain absolut terbesar untuk rata-rata).
- Mode pencarian cepat: learning_rate 0,05 sampai 0,1 dan 3 fold. 150 sampai 400 trial per study GBDT, 50 sampai 100 untuk NN.
- Konfirmasi: 3 konfigurasi terbaik dilatih ulang dengan learning_rate rendah (0,01 sampai 0,03) dan 5 fold penuh. Yang terbaik dibekukan ke `configs/best_params/<target>_<family>.yaml` dan dicatat di registry.
- Ruang pencarian awal ada di Lampiran H. Perlebar batas jika optimum menempel di tepi.
- Simpan plot parameter importance dan riwayat optimasi ke `reports/phases/hpo/`.
- Jika feature set berubah setelah HPO, lakukan tuning ulang ringan (50 trial, mulai dari parameter terbaik lama dengan `enqueue_trial`).

---

## 11. Fase 7: Multi-seed dan pemanfaatan data

- Seed bagging untuk konfigurasi teratas memakai `SEED_BAG`. Ukur gain bagging per keluarga (biasanya kecil untuk GBDT, besar untuk NN).
- Uji 10-fold untuk model final (`data/folds/folds_primary_k10_s42.parquet`): tiap model dilatih dengan 90% data dan OOF lebih stabil. Jika dipakai, bobot ensemble di-fit ulang di OOF 10-fold, dan semua komponen ensemble harus memakai file fold yang sama.
- Opsi full-fit (latih di 100% train dengan jumlah iterasi = median best_iteration x 1,1 sampai 1,2) tidak bisa divalidasi dengan CV. Pakai hanya jika didukung bukti public LB yang konsisten. Default tetap rata-rata model fold.

---

## 12. Fase 8: Ensemble, stacking, post-processing

Syarat: semua OOF berasal dari file fold yang sama dan diselaraskan berdasarkan `id`.

Per target, bandingkan metode berikut. Semua dievaluasi nested: parameter blending dipelajari di 4 fold OOF, dinilai di fold ke-5, diputar untuk semua fold.
1. Model tunggal terbaik.
2. Rata-rata top-N (N = 2 sampai 10).
3. Rata-rata berbobot: bobot tidak negatif dan berjumlah 1, dioptimasi dengan `scipy.optimize.minimize` (SLSQP) atau NNLS.
4. Hill climbing / Caruana ensemble selection (dengan pengembalian, inisialisasi top-3, 50 sampai 200 iterasi), di-bagging 20 kali dengan subset model acak ber-seed.
5. Ridge dengan `positive=True` dan intercept di atas OOF.
6. Stacking level-2: LightGBM dangkal (num_leaves 7 sampai 31, regularisasi kuat) atau Ridge di atas [OOF semua model untuk kelima target (cross-target) + top-20 fitur mentah + TE grup]. Ini sekaligus regressor chain: prediksi target lain ikut membantu target ini.
7. Level-2 MLP kecil (opsional).

Post-processing (nested): clip; kalibrasi linear per target; blend konsistensi t1 vs t3 x exposure (jika relasi fisik kuat).

Pilih metode terbaik per target berdasarkan skor nested. Jika selisih metode kompleks terhadap rata-rata berbobot kurang dari 0,1% relatif, pilih yang lebih sederhana. Rakit submisi 5 kolom, validasi, dan perbarui `BEST_CURRENT.csv` jika rata-rata CV_PRIMARY membaik. Catat ensemble di registry sebagai keluarga `ensemble` (ENS0001, ...) lengkap dengan daftar komponen dan bobot.

Laporkan juga skor ensemble di CV_SECONDARY sebagai ukuran ketahanan (latih ulang minimal 3 komponen berbobot terbesar di `folds_secondary`).

---

## 13. Fase 9: Teknik lanjutan (hanya jika didukung bukti)

- **Pseudo-labeling yang benar**: untuk fold f, teacher dilatih hanya di train-part fold f lalu memprediksi test; student dilatih di train-part + test berlabel semu (bobot 0,3 / 0,5 / 1,0), lalu dinilai di fold validasi f. Diterima hanya jika konsisten membaik.
- **Model residual**: GBDT kedua di atas residual ensemble dengan fitur grup (nested CV ketat).
- **Adversarial reweighting** jika ditemukan drift train-test.
- **Distilasi NN** dari ensemble.
- **Ide baru dari analisis error** (9.5): rumuskan hipotesis, uji, catat.

---

## 14. Fase 10: Strategi submit dan pemilihan final

- Anggaran 5 submisi per hari. Jika `KAGGLE_AUTO_SUBMIT: true`, agen memakai maksimal `MAX_AGENT_SUBMITS_PER_DAY`.
- Perintah: `kaggle competitions submit -c dicefest-2026-machine-learning -f <file> -m "<id> | CV <skor>"`, lalu `kaggle competitions submissions -c dicefest-2026-machine-learning` untuk membaca skor.
- Yang layak di-submit: (a) uji pembeda skema CV di Fase 2, (b) tonggak perbaikan CV yang berarti (minimal 0,5% relatif), (c) kandidat final.
- Setiap submisi tercatat di `submissions/submission_log.csv`. Setelah minimal 5 titik, hitung korelasi CV vs public LB dan tampilkan di dashboard.
- Jangan mengejar public LB. Public hanya 30% test (sekitar 55 ribu baris), selisih kecil adalah noise. Keputusan utama tetap berdasarkan CV_PRIMARY.
- Dua submisi final:
  - **Final A**: ensemble dengan CV_PRIMARY terbaik.
  - **Final B**: lindung nilai terhadap risiko terbesar yang teridentifikasi. Jika masih ada keraguan soal mekanisme split, B adalah ensemble terbaik menurut CV_SECONDARY atau yang tidak bergantung pada fitur grup. Jika tidak ada keraguan, B adalah varian dengan komposisi berbeda (lebih sederhana) dan CV hampir sama.
  - Tulis alasan di DECISIONS.md (D-FINAL) dan di NEEDS_USER.md, karena user harus memilih 2 submisi final di halaman Kaggle sebelum deadline.
- PENTING: file yang di-submit sebagai final harus persis output notebook final (Fase 11), bukan file hasil run eksperimen.

---

## 15. Fase 11: Audit reproducibility dan paket deliverable

### 15.1 Isi `final/`
- `final/src/` (salinan modul yang dibutuhkan, dibekukan), `final/configs/` (config final yang dibekukan), `final/run_all.py` (satu pintu: data → fitur → latih semua model final → ensemble → submisi).
- `final/DICEFEST2026_<TEAM_NAME>.ipynb`: notebook mandiri yang menjalankan pipeline yang sama dengan `run_all.py`. Sel pertama: cek versi library, set environment variable, path data yang mudah diubah. Sel terakhir: tulis submisi dan cetak sha256.
- `final/inference_from_artifacts.ipynb`: memuat model tersimpan di `final/artifacts/` (dengan checksum) dan menghasilkan submisi identik dengan cepat. Ini cadangan jika juri tidak bisa melatih ulang penuh.
- `final/requirements.txt` (versi dipin) dan `final/README_REPRODUCE.md` (langkah, hardware, waktu jalan per tahap, sha256 yang diharapkan).

### 15.2 Prosedur audit
1. Bekukan kode dan config pada `GATE_FREEZE_FINAL`. Setelah itu hanya perbaikan bug.
2. Run #1 dari kondisi bersih (hapus output final) → `submission_run1.csv`.
3. Run #2 dari kondisi bersih di proses baru → `submission_run2.csv`.
4. Bandingkan sha256 dan max |selisih|. Harus identik.
5. Eksekusi notebook secara headless: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 final/DICEFEST2026_<TEAM_NAME>.ipynb --output executed.ipynb`. Hasilnya harus identik dengan run #1.
6. Jika tidak identik: isolasi komponen (jalankan per model), perbaiki (pengaturan deterministik, versi CPU, thread tetap), lalu ulangi dari langkah 2.
7. Ukur dan catat waktu jalan. Jika sangat panjang (misal lebih dari 12 jam), pastikan notebook inference dari artefak siap dan terdokumentasi.
8. Minta subagent dengan konteks bersih mengaudit `final/` terhadap: kebocoran target, pemakaian `id`, data eksternal, seed yang belum dikunci, sumber nondeterminisme. Tindak lanjuti temuan yang mempengaruhi kebenaran hasil.
9. Submisi final ke Kaggle adalah file dari run notebook ini.

Catatan: hasil GPU bisa berbeda antar jenis GPU. Dokumentasikan hardware (RTX A4000, versi driver, CUDA) dan sediakan jalur inference dari artefak. Jika selisih skor versi CPU-only terhadap versi GPU kecil, pertimbangkan versi CPU untuk komponen yang tidak deterministik lintas hardware. Catat keputusan dan trade-off-nya.

---

## 16. Fase 12: Laporan akhir, writeup, presentasi

- `reports/FINAL_REPORT.md` (teknis lengkap): temuan data, bukti mekanisme split dan desain CV, transform target, ablation fitur, perbandingan model, HPO, komposisi dan bobot ensemble final, skor per target (CV_PRIMARY, CV_SECONDARY, public LB), tabel CV vs LB, bukti reproducibility (sha256), lingkungan, waktu jalan, dan apa saja yang tidak berhasil.
- `reports/WRITEUP_DRAFT.md` untuk PDF: Bahasa Indonesia akademik yang natural, tidak kaku, tidak terdengar seperti teks mesin, tanpa karakter em-dash (U+2014). Struktur: Abstrak; Pendahuluan dan pemahaman masalah (trilema transisi energi); Data dan eksplorasi; Strategi validasi; Rekayasa target dan fitur; Pemodelan; Ensemble; Hasil dan ablation; Interpretasi kebijakan; Reproducibility; Keterbatasan dan pengembangan; Kesimpulan. Cek ketentuan format writeup di https://dicefest.web.id bila dapat diakses; jika tidak, catat di NEEDS_USER.
- Interpretabilitas: SHAP per target pada sampel 20 sampai 50 ribu baris; agregasi |SHAP| per keluarga prefix (bld_dim, bld_era, hvac_dist, env_climate, socio_demog) per target; dependence plot fitur utama. Hubungkan ke tiga dilema kebijakan: skala vs intensitas termal (t1 vs t3), kemacetan gardu listrik (t2 dan t5), kesenjangan adopsi dan tenure (t4 dan t5).
- Figur 300 dpi di `reports/figures/`: distribusi target, perbandingan skema CV, ablation, perbandingan model, bobot ensemble, SHAP per keluarga, CV vs LB.
- `reports/PRESENTATION_OUTLINE.md`: alur 10 menit (8 sampai 10 slide) dengan catatan pembicara, plus 10 sampai 15 pertanyaan juri yang mungkin muncul beserta jawaban berbasis bukti (mengapa skema CV ini, bagaimana kebocoran dicegah, mengapa ensemble, bagaimana reproducibility dijamin, insight kebijakan apa yang didapat).
- Konversi writeup ke PDF bila pandoc tersedia. Jika tidak, Markdown + figur sudah cukup untuk diselesaikan user.

---

## 17. Jadwal dan gerbang (WIB)

| Kapan | Gerbang | Isi |
|---|---|---|
| 16 Okt malam | G1 | Fase 0 dan 1 selesai, D001 mekanisme split |
| 17 Okt malam | G2 | Fase 2 selesai (CV ditetapkan, baseline, determinisme, submisi pertama), Fase 3 berjalan |
| 19 Okt malam | G3 | Fase 3 dan 4 selesai: transform dan feature set per target ditetapkan |
| 21 Okt malam | G4 | Fase 5, 6, 7 selesai: zoo, HPO, multi-seed; ensemble v1 |
| 22 Okt 12:00 | G5 | Freeze kandidat final (Fase 8 dan 9 selesai). Mulai Fase 11 |
| 22 Okt malam | G6 | Dua run identik + notebook tereksekusi + draf writeup |
| 23 Okt paling lambat 12:00 | G7 | Paket final lengkap, submisi final ter-upload, NEEDS_USER berisi checklist akhir untuk user |

- Lebih cepat dari jadwal: pakai sisa waktu untuk HPO lebih dalam, seed tambahan, keragaman model, analisis error.
- Terlambat: lewati item eksploratif. Fase 11 tidak boleh dilewati.
- Di setiap gerbang tulis `reports/phases/GATE_<n>.md`: skor terbaik per target, apa yang selesai, risiko.

---

## 18. Jebakan yang harus dihindari
- Fit scaler, encoder, atau PCA yang memakai target di luar fold.
- Target encoding atau KNN-target tanpa cross-fitting.
- Bobot ensemble dipilih di OOF lalu skor OOF yang sama dilaporkan sebagai skor bersih (pakai evaluasi nested).
- Early stopping di fold validasi lalu skornya diklaim tanpa catatan (wajar untuk perbandingan relatif, tetapi sebut di laporan).
- Mencampur file fold antar eksperimen yang akan di-ensemble.
- Urutan kolom dari `set()`, merge tanpa kunci `id`, lupa mengurutkan submisi sesuai sample_submission.
- Training di skala log tetapi tidak mengevaluasi di skala asli, atau lupa koreksi bias back-transform.
- Lupa clip, atau prediksi negatif untuk target makro dan EUI.
- `time_limit` di pipeline final, AMP atau cuDNN benchmark tanpa uji determinisme.
- Dua job GPU berat bersamaan (OOM), atau terlalu banyak proses memuat data penuh (RAM 24 GB habis).
- Mencetak output raksasa ke percakapan.
- Menyerahkan file submisi yang bukan output notebook final.
- Mengubah file fold setelah eksperimen berjalan.

---

## 19. Komunikasi dan definisi selesai

Komunikasi: di setiap gerbang dan di akhir setiap fase, tampilkan ringkasan singkat di percakapan (maksimal 15 baris): skor terbaik per target dan rata-rata, apa yang berubah, langkah berikutnya, risiko. Detail tetap di file laporan.

Proyek dinyatakan selesai jika semua poin ini benar, dan buktinya ditampilkan di percakapan (evaluator `/goal` hanya membaca percakapan):
1. `reports/STATUS.md` memuat baris `STATUS: SEMUA FASE SELESAI`.
2. Dua run final dari kondisi bersih dan satu eksekusi notebook headless menghasilkan sha256 submisi yang sama. Tampilkan ketiganya.
3. File submisi final sudah di-submit ke Kaggle (atau tercatat di NEEDS_USER untuk di-submit user) dan sama persis dengan output notebook.
4. `final/` berisi notebook utama, notebook inference, requirements, README_REPRODUCE.md, artefak model + checksum.
5. `reports/FINAL_REPORT.md`, `reports/WRITEUP_DRAFT.md`, dan `reports/PRESENTATION_OUTLINE.md` ada dan lengkap.
6. `reports/COVERAGE.md` tidak memiliki item QUEUED atau RUNNING.
7. `reports/NEEDS_USER.md` berisi checklist akhir untuk user: memilih 2 submisi final di Kaggle, mengunggah PDF writeup dan notebook ke https://dicefest.web.id.

Jika waktu (`TZ=Asia/Jakarta date`) sudah melewati 23 Oktober 2026 20:00 WIB sebelum semua selesai: hentikan eksperimen baru, pastikan submisi terbaik yang valid dan paket final terbaik yang tersedia siap, lalu tulis kondisi apa adanya di STATUS.md dan NEEDS_USER.md.

---

## Lampiran A. Kolom `experiments/registry.csv`

`exp_id, parent_id, status, phase, created_wib, finished_wib, hypothesis, model_family, device, targets, target_transform, feature_set_id, n_features, cv_scheme, fold_file, seeds, config_path, params_hash, git_commit, data_hash, nrmse_mean, nrmse_t1, nrmse_t2, nrmse_t3, nrmse_t4, nrmse_t5, fold_nrmse_json, delta_vs_best_json, folds_improved_json, runtime_min, peak_ram_gb, peak_vram_gb, oof_path, test_pred_path, notes`

`nrmse_mean` hanya diisi jika kelima target dimodelkan. Kolom target yang tidak dimodelkan dibiarkan kosong.

## Lampiran B. Skema `metrics.json`

```json
{
  "exp_id": "E0042",
  "cv_scheme": "primary",
  "fold_file": "data/folds/folds_primary_k5_s42.parquet",
  "per_target": {
    "target_macro_gas_kwh": {
      "nrmse": "<float>",
      "rmse": "<float>",
      "fold_nrmse": ["<float>", "<float>", "<float>", "<float>", "<float>"],
      "best_iteration": ["<int>", "<int>", "<int>", "<int>", "<int>"]
    }
  },
  "nrmse_mean": null,
  "runtime_min": "<float>",
  "peak_ram_gb": "<float>",
  "peak_vram_gb": "<float>"
}
```

## Lampiran C. Template entri `reports/JOURNAL.md`

```markdown
### [2026-10-17 14:20 WIB] E0042-E0045 | Fase 3 | log1p vs ratio-to-exposure untuk t1
- Hipotesis:
- Setup: model, feature set, transform, CV, seed, perubahan dari parent
- Hasil: tabel NRMSE per target, delta vs terbaik, fold membaik x/5
- Kesimpulan: DITERIMA / DITOLAK, beserta alasannya
- Tindak lanjut:
```

## Lampiran D. Template `reports/DECISIONS.md`

```markdown
### D003 [2026-10-17 21:00 WIB] Transform target per target
- Konteks:
- Opsi yang dipertimbangkan:
- Bukti (exp_id, angka, file):
- Keputusan:
- Risiko dan mitigasi:
```

## Lampiran E. Template `reports/STATUS.md`

```markdown
# STATUS (diperbarui: <tanggal jam WIB>)
STATUS: BERJALAN
Fase aktif / gerbang berikutnya:
Job berjalan: lane | exp_id | PID | mulai | perkiraan selesai | log
Terbaik CV_PRIMARY: mean ... | t1 ... | t2 ... | t3 ... | t4 ... | t5 ... (sumber: exp/ens id)
BEST_CURRENT.csv: sumber, sha256, public LB
5 langkah berikutnya:
Risiko / blocker / NEEDS_USER:
```

## Lampiran F. Template config eksperimen

```yaml
exp_id: E0042
parent_id: E0031
phase: 3
hypothesis: "ratio-to-exposure dengan bobot E^2 menurunkan NRMSE t1"
targets: [target_macro_gas_kwh]
cv: {scheme: primary, fold_file: data/folds/folds_primary_k5_s42.parquet}
features: {set: FS0007}
target_transform: {type: ratio_exposure, exposure_col: "<kolom dari 5.E>", weight_power: 2}
model:
  family: lightgbm
  device: cpu
  params: {learning_rate: 0.05, num_leaves: 255}
seeds: [42]
postprocess: {clip: true, calibrate: none}
save_models: false
```

## Lampiran G. Validator submisi

1. Header persis sama dengan urutan resmi.
2. 183.987 baris; `id` integer, identik dan berurutan sama dengan `sample_submission.csv`.
3. Tidak ada NaN atau inf; semua kolom numerik.
4. t4 dan t5 di rentang 0 sampai 100; t1, t2, t3 tidak negatif.
5. Sanity distribusi: mean dan std tiap kolom dibandingkan dengan OOF dan target train. Beri peringatan jika selisih mean lebih dari 10%.
6. Tulis sha256 dan tambahkan baris ke `submissions/submission_log.csv`.

## Lampiran H. Parameter referensi dan ruang pencarian

LightGBM referensi (untuk baseline dan ablation):

```yaml
objective: regression
learning_rate: 0.05
num_leaves: 255
min_data_in_leaf: 100
feature_fraction: 0.6
bagging_fraction: 0.8
bagging_freq: 1
lambda_l2: 1.0
max_bin: 255
n_estimators: 20000
early_stopping_rounds: 300
seed: 42
deterministic: true
force_row_wise: true
verbose: -1
```

Ruang pencarian awal:
- **LightGBM**: num_leaves 31-2047 (log), max_depth -1 atau 6-16, min_data_in_leaf 5-1000 (log), min_sum_hessian_in_leaf 1e-3 sampai 10 (log), feature_fraction 0.2-1.0, bagging_fraction 0.5-1.0, lambda_l1 1e-8 sampai 10 (log), lambda_l2 1e-8 sampai 100 (log), min_gain_to_split 0-1, max_bin {63, 127, 255, 511, 1023}, path_smooth 0-20, extra_trees {false, true}, linear_tree {false, true} dengan linear_lambda 0-10.
- **XGBoost**: max_depth 4-14 (depthwise) atau max_leaves 31-1023 (lossguide), min_child_weight 1e-2 sampai 300 (log), subsample 0.5-1.0, colsample_bytree 0.2-1.0, colsample_bynode 0.5-1.0, reg_alpha 1e-8 sampai 10 (log), reg_lambda 1e-3 sampai 100 (log), gamma 0-5, max_bin {256, 512, 1024}.
- **CatBoost**: depth 4-12, l2_leaf_reg 1-50 (log), random_strength 1e-3 sampai 10 (log), bootstrap_type Bayesian (bagging_temperature 0-5) atau Bernoulli (subsample 0.5-1.0), border_count {128, 254}, grow_policy {SymmetricTree, Depthwise, Lossguide} dengan min_data_in_leaf dan max_leaves untuk yang non-symmetric.
- **NN (MLP, TabM, RealMLP)**: n_blocks 1-5, d_block 128-2048, dropout 0-0.4, lr 1e-4 sampai 3e-3 (log), weight_decay 1e-6 sampai 1e-2 (log), batch 512-8192, embedding numerik {none, periodic, piecewise-linear}, n_frequencies 16-96, frequency_scale 0.01-10 (log), dimensi embedding ID 8-32.
- Learning rate: pencarian di 0,05 sampai 0,1; model final di 0,01 sampai 0,03 dengan early stopping.
