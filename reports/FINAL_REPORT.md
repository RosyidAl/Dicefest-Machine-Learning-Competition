# FINAL REPORT (teknis): DICE FEST 2026 Machine Learning

Laporan teknis lengkap untuk audit. Writeup yang lebih ringkas ada di `reports/WRITEUP_DRAFT.md`. Sumber kebenaran angka: `experiments/registry.csv`, `reports/JOURNAL.md`, `reports/DECISIONS.md`. Angka bertanda [[...]] diisi setelah run audit selesai.

## 1. Ringkasan
- Submisi final (Final A): blend v16, 34 anggota, koreksi residual grup + kNN kemiripan, model adopsi binomial, salin 77 baris duplikat (D006, D007).
- CV_PRIMARY nested (ENS0018): t1 0,24524 | t2 0,31608 | t3 0,65101 | t4 0 | t5 0,0357 | rata-rata 0,24960. Dengan koreksi residual (estimasi holdout) sekitar 0,2489.
- Public LB: 0,24943 (sub_v16dev, sha256 e9375ba6...9a7f).
- Reproducibility: run #1 (`final/run_all.py`) dan notebook headless memberi sha256 [[SHA]].

## 2. Data dan temuan
Rincian lengkap di `reports/eda/EDA_REPORT.md` (Fase 1). Yang menentukan pemodelan:
- Identitas eksak: EUI x `bld_dim_06` = gas (selisih relatif maks 7e-16); rasio bangunan bersih = round(100 x `bld_dim_02`/`bld_dim_01`, 2) di 100% baris.
- Adopsi pemanas listrik konstan per subzona (unit sensus OA). 113 dari 132 kolom sensus konstan per subzona. Jumlah rumah tangga N dapat direkonstruksi eksak dari pecahan `hvac_dist`.
- `sub_n` (nomor subzona) berurut secara spasial: 99,07% pasangan subzona berurutan berada di zona yang sama. `env_climate_01/03/05` mendefinisikan sel grid iklim (sekitar 15 baris train per sel).
- Noise label: varians residual log = a + b/M. Blend v16: gas a = 0,0093, b = 0,367 (73% varians residual adalah noise); listrik a = 0,0116, b = 0,577 (77%). Lantai skor dari fitur saja sekitar 0,21 (`reports/noise_floor.csv`, Gambar 2).

## 3. Mekanisme split dan desain CV (D001, D002)
| Bukti | Angka |
|---|---|
| Rasio baris train | 0,700000 |
| Sebaran test per zona | binomial, std z-score 0,989 |
| Baris test di subzona tak terlihat | 8,14% (harapan split acak per baris 8,17%) |
| Adversarial validation | AUC 0,4997 tanpa ID, 0,4990 dengan ID |

- CV_PRIMARY: KFold(shuffle, seed 42), 5 fold untuk screening dan 10 fold untuk anggota final (`data/folds/folds_primary_k*_s42.parquet`).
- CV_SECONDARY: GroupKFold per `zone_region_id` (`data/folds/folds_secondary_k5_s42.parquet`). [[Hasil E0271-E0278.]]
- Validasi CV terhadap LB (Gambar 3, `reports/cv_lb_history.csv`): pada 10 submisi, public LB = CV nested + estimasi holdout post-processing + 0,00052 (sd 0,00004).

## 4. Formulasi target
| Kode | Formulasi | Dipakai untuk |
|---|---|---|
| T0 | identitas | baseline |
| T1 | log + koreksi bias | gas, listrik (keluarga log) |
| T4 | rasio terhadap eksposur (meter, luas lantai, luas selubung, jumlah sertifikat), bobot agar loss setara MSE skala asli | sebagian besar anggota |
| T5 | Poisson pada laju per eksposur | diuji (E0260-E0265, EM0002-EM0005), tidak masuk final (D007) |
| T7 | proporsi + cross-entropy | adopsi (binomial cacah kategori hilang) |
| T10 | gas = EUI x luas | keluarga gas |

## 5. Fitur
| Blok | Isi | Bukti |
|---|---|---|
| FB01 | buang kolom konstan / duplikat (11 kolom) | EDA |
| FB02 | ID spasial numerik (`zone_n`, `sub_n`) | |
| FB04 | rasio fisik (luas per meter, luas per sertifikat, meter per sertifikat, GEA/luas, dll.) | fitur SHAP teratas |
| FB05 | agregat transduktif per subzona/zona dari train + test (tanpa target) | LGB EUI -0,0011 |
| FB06 | TE out-of-fold: subzona, zona, jendela `sub_n` ±2/±10, sel iklim di luar subzona sendiri | sel: XGB EUI -0,0012 |
| FB07 | kNN-target OOF dalam ruang stok bangunan (W/k = 30/20, 100/20 spasial, 300/50) | E0190 -> E0191: gas -0,0011, EUI -0,0022 (5/5 fold) |
| LK1 | uji kebocoran blok bertarget (`tests/test_no_leakage.py`) | lulus |

## 6. Model
Ringkasan anggota final per keluarga: `reports/WRITEUP_DRAFT.md` Tabel 1 dan `reports/blend_family_share.csv`.
- XGBoost CUDA hist (17 anggota), LightGBM CPU deterministic (7), MLP PyTorch polos / PLR / TabM-mini (8), MLP multi-task gas + listrik (1 jaringan, 2 anggota).
- Baseline Fase 2 (CV_PRIMARY 5-fold, rata-rata 5 target): B0 rata-rata 1,0002; B1 Ridge 0,4001; B4 XGBoost 0,2781; B6 MLP multi-output 0,2970. B2 dan B5 terhenti oleh insiden RAM 16 Okt dan tidak diulang (registry: failed).
- Determinisme (D004): XGBoost CUDA, LightGBM, dan semua MLP identik byte antar run; CatBoost GPU tidak, sehingga dikeluarkan (biaya CV sekitar 0,00001).

## 7. Ensemble dan post-processing
- Blend NNLS per kolom yang dinilai (gas dengan skala luas, EUI, listrik), seleksi anggota greedy maju-mundur dengan skor nested (`exp/analysis/select_v16.py`). Daftar anggota dan bobot: `final/configs/members.yaml`, `final/work/v16_dev/summary.json`.
- Tidak lebih baik dari NNLS: stacking level-2 LightGBM, blend per kelompok ukuran, bobot dari holdout, blend khusus subzona baru (selisih <= 0,00006).
- Koreksi residual (PP0001 + PP0004): rata-rata residual OOF yang di-shrink dari zona, sel iklim, jendela ±30 subzona, dan 5-6 konfigurasi kNN kemiripan per metrik. λ di-fit di holdout P70/H30 (`exp/resid_knn_v16.json` = `final/configs/resid_knn.json`). Gain holdout: gas 0,24643 -> 0,24539, EUI 0,65115 -> 0,64961, listrik 0,31508 -> 0,31477.
- Uji kepadatan (17 Okt): gain koreksi naik dengan kepadatan kumpulan tetangga, dan λ dari kumpulan yang lebih jarang hampir optimal (rugi <= 0,00004 per metrik). Estimasi holdout bersifat konservatif untuk test.
- Adopsi (PP0002): nilai train disalin untuk subzona terlihat; subzona baru memakai LightGBM cross-entropy atas proporsi kategori hilang dengan bobot cacah.
- Duplikat (PP0003): 77 baris test menyalin target baris train yang identik pada kunci (socio_demog_005, subzona, M, E, A).

## 8. Skor per target
| Model | t1 gas | t2 listrik | t3 EUI | t4 | t5 | rata-rata | public LB |
|---|---|---|---|---|---|---|---|
| ENS0017 v15nc | 0,24591 | 0,31641 | 0,65211 | 0 | 0,0357 | 0,25002 | 0,24954 (dengan koreksi kNN) |
| ENS0018 v16 | 0,24524 | 0,31608 | 0,65101 | 0 | 0,0357 | 0,24960 | 0,24943 |

## 9. Yang tidak berhasil
| Ide | Hasil | Catatan |
|---|---|---|
| Objektif Poisson sebagai anggota (EM0002-EM0005) | blend -0,000035 | di bawah aturan 0,1% (D007) |
| Loss Poisson pada MLP (E0266-E0270) | EUI +0,0045 | ditolak |
| Koreksi residual lintas target | gas +0,00001, EUI +0,00002, listrik -0,00003 | ditolak |
| Adopsi sebagai fitur gas/listrik | korelasi residual <= 0,01 | ditolak |
| FB07b: kNN-target tambahan (E0279-E0280) | XGB gas -0,00031 (5/5), EUI -0,00045 (4/5, SE 0,00048) | ditolak, perkiraan blend -0,00003 |
| λ koreksi diskalakan (kepadatan) / interaksi ukuran | <= -0,00001 / memburuk | ditolak |
| 20 fold vs 10 fold | OOF membaik, holdout tidak | |
| CatBoost di final | tidak deterministik | D004 |
| Pencarian kebocoran (urutan ID, gas "kembar", rasio antar kolom, copula) | tidak ada sinyal | hanya 77 duplikat |

## 10. Reproducibility dan lingkungan
- `final/run_all.py`, notebook `final/DICEFEST2026_[[TEAM_NAME]].ipynb`, notebook cadangan `final/inference_from_artifacts.ipynb`, `final/README_REPRODUCE.md`, `final/requirements.txt`.
- Uji drift (16 Okt): anggota yang dilatih ulang lewat pipeline final identik byte dengan array pengembangan; inference dari model tersimpan identik byte.
- Run #1: [[waktu per jalur, sha256]]. Notebook headless: [[waktu, sha256]].
- Hardware: NVIDIA RTX A4000 16 GB, Intel Xeon Silver 4216 16 vCPU, RAM 32 GB, Ubuntu 26.04, Python 3.12.
