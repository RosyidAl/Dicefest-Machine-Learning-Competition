# COVERAGE: semua pendekatan di MASTER_PROMPT dan statusnya

Status yang dipakai:
- **DONE**: ada exp_id dan hasilnya.
- **SKIPPED**: ada alasan berbasis bukti.
- **RUNNING** / **QUEUED**: belum selesai.

Exp_id E0001 sampai E0168, ENS dan PP adalah run lama (12 sampai 16 Okt) yang di-backfill dari array OOF (scripts/backfill_registry.py).

Catatan delta: "H" = holdout P70/H30 (seed 7). Delta skor total = delta per-target / 5.

## Validasi
| ID | Pendekatan | Status | Bukti / hasil |
|---|---|---|---|
| V01 | CV_PRIMARY meniru split test (KFold acak per baris) | DONE | D001/D002; CV vs LB konsisten (offset +0,0000 sampai +0,0005 di 9 submisi) |
| V02 | CV_SECONDARY GroupKFold per zone_region_id dilaporkan | QUEUED | |
| V03 | Adversarial validation train vs test | DONE | LGB 5-fold AUC 0,4997 (tanpa ID) dan 0,4990 (dengan ID); EDA_REPORT 5.B |
| V04 | Uji pembeda skema CV via LB (dengan vs tanpa info grup) | QUEUED | |
| V05 | Kriteria penerimaan 4/5 fold atau 2 SE | DONE | dipakai di E0191 (5/5 fold), E0264/E0265 (5/5 fold) |

## Baseline (Fase 2)
| ID | Pendekatan | Status | Bukti / hasil |
|---|---|---|---|
| B0 | Prediksi rata-rata | DONE | E0169 (primer 1,00016), E0179 (sekunder 1,0006) |
| B1 | Ridge | DONE | E0170 (primer 0,40011), E0180 (sekunder 0,40136) |
| B2 | LightGBM referensi tanpa ID spasial | SKIPPED | E0171 terhenti insiden RAM 16 Okt; digantikan E0271-E0278 (LGB dengan/tanpa TE, CV primer dan sekunder) |
| B3 | B2 + ID spasial (numerik, kategori) | SKIPPED | idem B2; ID spasial numerik dipakai di semua anggota |
| B4 | XGBoost GPU default | DONE | E0173 (0,27808) |
| B5 | CatBoost GPU + cat_features ID | SKIPPED | E0174 terhenti insiden RAM; CatBoost GPU tidak deterministik (D004) |
| B6 | MLP multi-output sederhana | DONE | E0175 (0,29699) |

## Determinisme per keluarga
| ID | Keluarga | Status | Bukti / hasil |
|---|---|---|---|
| R01 | LightGBM CPU deterministic | DONE | rerun lgb_epm_wM_tew identik byte (14 Okt) |
| R02 | XGBoost GPU hist | DONE | rerun zoo1/zoo3/zoo5/grid identik byte (14 Okt) |
| R03 | CatBoost GPU | DONE | TIDAK deterministik (c4_lepa: iterasi beda, rms beda pred 352 kWh) |
| R04 | MLP polos (PyTorch deterministic) | DONE | rerun n4f_lepm identik byte |
| R05 | MLP PLR, TabM-mini, MLP multi-task | DONE | hash identik di 2 proses (logs/repro_check_repro_nn.result), D004 |

## Transform target (Fase 3)
| ID | Transform | Status | Bukti / hasil |
|---|---|---|---|
| T0 | identity | DONE | x4_gas, x4_eui_wA, x4_elec, la_gas, la_elec (E0xxx, registry) |
| T1 | log1p + koreksi bias | DONE | leui, lgpm, lepm, lepa; smearing hanya -0,00004 |
| T2 | sqrt | QUEUED | |
| T3 | Yeo-Johnson | QUEUED | |
| T4 | rasio terhadap exposure + bobot E^p | DONE | gpm, gpgea, gpn, epm, epa dengan bobot 1/A, M, 1/M; anggota terbaik |
| T5 | Tweedie / Gamma / Poisson | DONE | E0260-E0265: Poisson menang di model tunggal, Tweedie 1,2 kalah; anggota EM0002-EM0005 hanya -0,000035 di blend (D007); MLP Poisson ditolak (E0266-E0270) |
| T6 | Huber | QUEUED | |
| T7 | proporsi + cross_entropy (t4, t5) | DONE | t5: adopt_v2 (LGB cross_entropy, model binomial); t4 identitas eksak |
| T8 | logit (t4, t5) | QUEUED | |
| T9 | hurdle | QUEUED | |
| T10 | target turunan t1 = t3 x A | DONE | identitas EUI = gas / bld_dim_06 eksak; seluruh keluarga gas memakai ini |
| PP1 | clip rentang valid | DONE | semua submisi |
| PP2 | kalibrasi linear OOF | DONE | intercept bebas di blend: <= 0,00006 di H |

## Feature engineering (Fase 4)
| ID | Blok | Status | Bukti / hasil |
|---|---|---|---|
| FB01 | Buang kolom konstan / duplikat | DONE | common.constant_cols |
| FB02 | ID spasial numerik / kategori / frequency | DONE | zone_n, sub_n numerik; count via agg_features |
| FB03 | Komposisi: entropi, HHI, dominan, indeks era | QUEUED | |
| FB04 | Fisika energi (rasio luas, GEA/GIA, dll) | DONE | common.base_features (area_per_meter, gea_ratio, dst) |
| FB05 | Agregat grup transductive | DONE | Data.agg_features: LGB gpm EUI -0,0011 |
| FB06 | Target encoding OOF (subzone, zone, window sub_n, sel iklim) | DONE | te.py; sel iklim EUI -0,0012, window, 10-fold |
| FB07 | KNN-target OOF | DONE | E0190-E0191 (gas -0,0011, EUI -0,0022, 5/5 fold), EM0001, anggota xk_/lk_/pk_/mk1 di v16 (ENS0018) |
| FB08 | PCA / SVD transductive | QUEUED | |
| FB09 | KMeans cluster + TE | QUEUED | |
| FB10 | Pencarian interaksi | QUEUED | |
| FB11 | Transform untuk NN/linear | DONE | nn.preprocess: log1p kolom miring + standardisasi; PLR embedding |
| FB12 | Seleksi fitur (null importance) | QUEUED | |
| LK1 | Tes kebocoran blok ber-target (tests/test_no_leakage.py) | DONE | TE grup (train_lib) dan FB07 knn_target lulus: fitur validasi/test identik setelah label validasi diacak |

## Model zoo (Fase 5)
| ID | Model | Status | Bukti / hasil |
|---|---|---|---|
| M01 | LightGBM gbdt | DONE | l4_*, la_*, lgb_* (registry) |
| M02 | LightGBM linear_tree | DONE | H: tidak ada gain blend |
| M03 | LightGBM extra_trees | DONE | H: tidak ada gain blend |
| M04 | LightGBM GOSS | QUEUED | |
| M05 | ID spasial kategori vs TE | QUEUED | |
| M06 | XGBoost depthwise | DONE | x4/x8/xa (d8, d10) |
| M07 | XGBoost lossguide | DONE | xgb_eui_lg, xgbl_* |
| M08 | XGBoost max_bin 1024 | QUEUED | |
| M09 | XGBoost kategori native | QUEUED | |
| M10 | CatBoost GPU | DONE | c4_*, cat_* (tidak deterministik, keluar dari final nc) |
| M11 | CatBoost depth 10 / grow policy / border | DONE (sebagian) | depth 10: tidak ada gain di H; grow policy QUEUED |
| M12 | MLP + embedding numerik PLR | DONE | p1/p2/p3 |
| M13 | MLP loss metrik langsung | QUEUED | |
| M14 | TabM | DONE | t1_* (TabM-mini + PLR) |
| M15 | RealMLP (pytabkit) | QUEUED | |
| M16 | pytabkit TD GBDT defaults | QUEUED | |
| M17 | CatBoost MultiRMSE | QUEUED | |
| M18 | XGBoost multi_output_tree | QUEUED | |
| M19 | GPBoost | QUEUED | |
| M20 | KNN regressor | QUEUED | |
| M21 | ExtraTrees | QUEUED | |
| M22 | Ridge fitur kaya | QUEUED | |
| M23 | AutoGluon | QUEUED | |
| M24 | FT-Transformer | QUEUED | |
| M25 | ResNet-MLP | QUEUED | |
| M26 | Foundation model tabular lokal (TabPFN, TabICL, dst) | QUEUED | |
| M27 | H2O AutoML / FLAML | QUEUED | |
| M28 | Distilasi NN | QUEUED | |
| M29 | MLP multi-task (gas + elec) | DONE | m1: EUI tunggal terbaik 0,66077; blend -0,00004 |
| EA1 | Analisis error residual | DONE | per ukuran M, per grup, segmen unseen; tidak ada segmen tersisa |

## HPO (Fase 6)
| ID | Pendekatan | Status | Bukti / hasil |
|---|---|---|---|
| H01 | Optuna TPE per (target, keluarga) | QUEUED | tuning manual/grid sudah DONE: exp_xgb_grid, exp_htune (XGB d10 lr .03) |
| H02 | Konfirmasi lr rendah 5-fold | DONE | x8_* (d10, lr .03) |

## Multi-seed dan data (Fase 7)
| ID | Pendekatan | Status | Bukti / hasil |
|---|---|---|---|
| S01 | Seed bagging NN | DONE | 3 seed; 6 seed = 3 seed |
| S02 | Seed bagging GBDT | QUEUED | |
| S03 | 10-fold | DONE | XGB EUI 0,6677 ke 0,6659 |
| S04 | Full-fit | DONE | lebih buruk untuk MLP (H) |

## Ensemble (Fase 8)
| ID | Metode | Status | Bukti / hasil |
|---|---|---|---|
| EN1 | Model tunggal terbaik | DONE | registry |
| EN2 | Rata-rata top-N | QUEUED | |
| EN3 | Rata-rata berbobot NNLS | DONE | ENS0001 sampai ENS0017 (nested honest) |
| EN4 | Hill climbing / Caruana | QUEUED | |
| EN5 | Ridge positive + intercept | DONE (sebagian) | intercept bebas di H <= 0,00006 |
| EN6 | Stacking level-2 LGB | DONE | tidak ada gain (stack.py) |
| EN7 | Level-2 MLP | QUEUED | |
| EN8 | Ensemble di CV_SECONDARY | QUEUED | |
| EN9 | Blend per bin ukuran | DONE | tidak ada gain |

## Teknik lanjutan (Fase 9)
| ID | Pendekatan | Status | Bukti / hasil |
|---|---|---|---|
| A01 | Pseudo-labeling | QUEUED | |
| A02 | Model residual (koreksi grup + kNN) | DONE | PP0001, PP0004; LGB residual tahap dua hanya -0,00006 |
| A03 | Adversarial reweighting | SKIPPED | AUC adversarial 0,50: tidak ada drift yang bisa dikoreksi (V03) |
| A04 | Distilasi NN dari ensemble | QUEUED | lihat M28 |
| A05 | Ide dari analisis error | DONE | adopt_v2 (N rumah tangga eksak), dup_fix, koreksi kNN |
