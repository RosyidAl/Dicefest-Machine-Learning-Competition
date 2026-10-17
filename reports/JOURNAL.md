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

### [2026-10-17 00:55 WIB] LB | sub_v16dev (kandidat final A)
- Public LB 0,24943 (user), vs sub_v15ncrkd 0,24954: -0,00011.
- Holdout memperkirakan -0,00054. Arah sama, besarnya lebih kecil. Public LB hanya sekitar 55 ribu baris, jadi selisih sekecil ini masih dalam rentang noise; keputusan tetap memakai CV/holdout.

### [2026-10-17 02:25 WIB] E0260-E0263 | Fase 3 | A/B Poisson vs formulasi rasio terbaik (LGB lr 0,1, 3 fold, FS_legacy + agg + TE)
- Hipotesis: objective Poisson mengalahkan formulasi rasio MSE yang dipakai anggota kita (E0260).
- Hasil (NRMSE OOF 3 fold):

  | Exp | Formulasi | gas | EUI | elec |
  |---|---|---|---|---|
  | E0260 | rasio per meter, MSE (bobot M^2/A gas, M elec) | 0,25672 | 0,67119 | 0,32633 |
  | E0261 | Poisson pada nilai mentah | 0,25468 | 0,66755 | 0,32543 |
  | **E0262** | **Poisson pada laju per meter, bobot M (offset log M)** | **0,25310** | **0,66536** | **0,32385** |
  | E0263 | Tweedie 1,2 pada laju per meter, bobot M | 0,25629 | 0,67030 | 0,32587 |

- E0262 vs E0260: gas -0,0036, EUI -0,0058, elec -0,0025; membaik di 3/3 fold untuk gas dan elec (fold-wise: 0,25200/0,25429/0,25300 vs 0,25556/0,25800/0,25659).
  Melampaui SE fold, memenuhi kriteria 6.2.
- Tweedie 1,2 hampir tidak lebih baik dari MSE dan berhenti jauh lebih awal (it sekitar 260 vs 800); Poisson murni yang bekerja.
- Kesimpulan: DITERIMA untuk model tunggal. Formulasi GLM laju dengan exposure meter cocok dengan mekanisme noise (rata-rata rumah tangga, varians sebanding mean).
- Tindak lanjut: versi XGB (E0264 gpm, E0265 EUI) sedang jalan; bila menang, buat anggota Poisson full 10-fold + holdout (xk/lk dengan fitur kNN) untuk kandidat v17 dan ukur gain blend + koreksi di holdout.

### [2026-10-17 03:40 WIB] E0264-E0265 | Fase 3 | hasil Poisson XGB
- Hasil vs E0190 (XGB MSE rasio, 5-fold):

  | Exp | gas | EUI | fold membaik |
  |---|---|---|---|
  | E0190 MSE gpm (bobot M^2/A) | 0,25291 | 0,66444 | |
  | E0264 Poisson gpm (bobot M) | 0,25085 (-0,0021) | 0,66054 (-0,0039) | 5/5 gas, 5/5 EUI |
  | E0265 Poisson EUI (bobot A) | 0,25043 (-0,0025) | 0,66307 (-0,0014) | 5/5 gas, 5/5 EUI |

- Iterasi sekitar 3500 (vs 1650 MSE), waktu sekitar 2x.
- Kesimpulan: DITERIMA. E0264 tanpa fitur kNN dan 5-fold sudah mendekati xk_gpm (EUI 0,65977, dengan kNN dan 10-fold).
  E0264 unggul di EUI, E0265 unggul di gas, jadi keduanya dibuat sebagai anggota (keragaman untuk blend).
- Tindak lanjut: anggota v17 (src/exp_v17_pois.py, fitur v16 + kNN, 10-fold + holdout): xp_gpm, xp_eui, xp_epm (GPU, rantai logs/v17_pois_chain.sh mulai 03:38);
  lp_gpm, lp_epm (LGB CPU) menunggu jalur CPU run #1 selesai. Lalu holdout: fit ulang λ koreksi, seleksi anggota, keputusan v17.

### [2026-10-17 09:00 WIB] Insiden kedua: semua job mati serentak sekitar 04:05 WIB
- Run #1 (10 dari 34 anggota selesai), rantai anggota Poisson (xp_gpm fold 2) dan skrip penjaganya mati bersamaan antara 04:02 dan 04:06 WIB.
- Bukan OOM: container tidak restart (uptime 8:55), memory.events cgroup oom = 0, rantai bash kecil ikut mati tanpa sempat menulis rc.
  Penyebab pasti tidak terlihat di journal; dugaan terkuat pembersihan proses di cgroup code-server saat sesi klien terputus.
  Insiden 16 Okt 20:40 yang dikira OOM kemungkinan sama.
- Tindakan: semua job kini dijalankan sebagai unit systemd terpisah (systemd-run --unit=dice-run1 / dice-v17), di luar cgroup terminal.
  Run #1 dilanjutkan tanpa --clean: 10 anggota yang sudah selesai dipakai ulang (anggota deterministik; tiap anggota dilatih di proses baru). Audit bersih penuh tetap dilakukan lewat eksekusi notebook headless.

### [2026-10-17 09:20 WIB] E0266-E0270 | Fase 3/5 | loss Poisson pada PLR MLP (fold 0)
- Hipotesis: loss deviance Poisson (link log) juga memperbaiki MLP PLR seperti pada GBM (E0262/E0264), sehingga anggota NN Poisson memberi keragaman baru.
- Setup: setelan pk_* (PLR, 1024-512-256, 12 epoch, epoch tetap 11), input v16 + kNN, 1 seed, fold 0 dari split 10-fold; src/exp_nn_pois_ab.py.
  Opsi baru loss="poisson" di src/nn.py (default "mse" tidak mengubah anggota lama).
  - E0266 MSE gpm (bobot M^2/A, acuan) vs E0267 Poisson gpm (bobot M) vs E0268 Poisson EUI (bobot A)
  - E0269 MSE epm vs E0270 Poisson epm (bobot M)

### [2026-10-17 09:25 WIB] E0266-E0270 | hasil loss Poisson pada PLR MLP (fold 0, 1 seed)
| Exp | Formulasi | EUI | gas | elec |
|---|---|---|---|---|
| E0266 | MSE gpm (acuan) | 0,66687 | 0,25022 | |
| E0267 | Poisson gpm | 0,67138 | 0,25086 | |
| E0268 | Poisson EUI | 0,67086 | 0,25096 | |
| E0269 | MSE epm (acuan) | | | 0,31969 |
| E0270 | Poisson epm | | | 0,32054 |
- Kesimpulan: DITOLAK. Untuk NN, MSE dengan bobot yang sudah disetel tetap lebih baik; gain Poisson khusus GBM (kemungkinan karena GBM MSE memboroskan kapasitas pohon pada baris besar yang noisy).

### [2026-10-17 09:25 WIB] Analisis | pencarian sinyal baru (semua negatif)
- Kategori pemanas yang hilang r (dari target adopsi, diketahui persis di 92% baris test): korelasi dengan residual log gas -0,0003, log elec -0,0068, EUI 0,0037. Tidak dipakai.
- Hipotesis Gaussian copula: ridge pada skor normal semua fitur, R2 ruang-z gas 0,907 / elec 0,874 / EUI 0,464 vs blend 0,929 / 0,892 / 0,587; residual berkorelasi 0,88 sampai 0,93 dengan residual blend. Tidak ada sinyal linear-copula tersisa.
- Gas kembar (4259 nilai, 8618 baris train, identik sampai presisi penuh float): 186 baris adalah duplikat penuh (sudah ditangani dup_fix).
  Sisanya berada di subzone/zone berbeda dan tidak berbagi fitur di luar efek modus kolom (bld_dim_39/27/42/29 dst. sama karena hampir konstan).
  Artefak pembangkitan data, tidak bisa diidentifikasi dari fitur.
- Kolom bernilai hampir unik (socio_demog_005, bld_dim_17): bukan koordinat, tidak ada identitas rasio/produk.
- Baris dengan nilai fitur identik (kolom apa pun): korelasi residual leave-one-out maksimum 0,09 (1.270 baris).

### [2026-10-17 09:45 WIB] xp_gpm (anggota v17) | gain blend
- Model tunggal (10-fold, input v16 + kNN, Poisson gpm bobot M): EUI 0,65741, gas 0,24899. Terbaik sejauh ini (xk_gpm MSE: 0,65977 / 0,25042).
- Seleksi (exp/analysis/select_v17.py, blend honest nested): v16 0,242467 -> 0,242451 (gas+EUI 0,896258 -> 0,896178), lk_gpm terbuang.
- Kesimpulan: gain model tunggal hampir tidak menular ke blend (-0,000016 total). Ensemble MSE/log sudah menyerap efek yang sama. Tunggu anggota Poisson elec dan holdout sebelum memutuskan v17.
- Catatan: GPU tanpa kontensi (jalur GPU run #1 dibekukan 09:24) memotong waktu fold XGB dari sekitar 600 s menjadi sekitar 100 s.

### [2026-10-17 11:00 WIB] xp_eui + xp_epm (anggota v17) | gain blend
- Model tunggal: xp_eui EUI 0,66008 / gas 0,24873 (Poisson pada EUI, bobot A); xp_epm elec 0,31977 (Poisson listrik per meter).
- Seleksi maju-mundur (exp/analysis/select_v17_r2.log): +xp_gpm, +xp_eui (gas+EUI 0,896258 -> 0,896161), +xp_epm (elec 0,316075 -> 0,316025), x4_elec dan xgb_epm_wM_tew_k10 terbuang.
- Total blend honest 0,242467 -> 0,242437 (-0,00003). Sama seperti xp_gpm: efek Poisson sudah terserap ensemble.
- Uji ulang adopsi sebagai fitur (subzone seen): adopsi rata-rata 5,38 vs sisa pangsa hvac 6,18 (r = 0,80, sd 1,17), korelasi dengan residual blend |r| <= 0,01. Konsisten dengan entri 09:25, tidak dipakai.
- Berikutnya: xp_gpgea / xp_epa / xp8_gpm (dice-v17b, GPU), uji koreksi residual lintas-target di holdout (dice-xt), lalu fit holdout v17.

### [2026-10-17 11:15 WIB] Koreksi residual lintas target + keputusan v17
- Hipotesis: residual listrik tetangga membantu koreksi gas/EUI (dan residual gas membantu listrik), karena residual log gas dan listrik berkorelasi 0,24 per baris dan 0,69 antar rata-rata zona.
- Setup (scratchpad xtarget.py, holdout P70/H30): fitur koreksi sendiri persis resid_knn.py (grup + kNN), ditambah rata-rata residual relatif keluarga lain (zona, sel, jendela, 4 konfigurasi kNN) dikali prediksi baris. λ cross-fitted 5-fold di H, bootstrap 200x.
- Hasil (gain di atas koreksi sendiri): gas +0,00001 (sd 0,00001), EUI +0,00002 (sd 0,00003), elec -0,00003 (sd 0,00003).
- Kesimpulan: DITOLAK. Faktor bersama gas-listrik sudah tertangkap oleh koreksi masing-masing.
- Efek sampingan: rantai holdout dengan anggota Poisson (xp_gpm, xp_eui, xp_epm ikut di blend holdout) memberi gas 0,24538 / EUI 0,64959 / elec 0,31475 vs v16 0,24539 / 0,64961 / 0,31477.
- xp_gpgea (full): EUI 0,65985, gas 0,24911. Seleksi dengan empat anggota Poisson: blend honest -0,000035.
- Keputusan D007: v16 tetap Final A. Rantai v17 dihentikan 11:11, jalur GPU run #1 dilanjutkan (SIGCONT).

### [2026-10-17 11:15 WIB] Fase 12 | figur dan SHAP
- src/make_figures.py -> reports/figures/fig1_targets, fig2_noise_floor, fig3_cv_lb, fig4_blend_weights, fig5_gains (300 dpi).
  Data pendukung: reports/cv_lb_history.csv, reports/ablation_summary.csv, reports/noise_floor.csv, reports/blend_family_share.csv.
- Noise floor pada blend v16: gas a = 0,0093, b = 0,367 (73% varians residual adalah noise rata-rata rumah tangga); elec a = 0,0116, b = 0,577 (77%).
- CV vs LB (10 submisi): public LB = CV nested + estimasi holdout post-processing + 0,00052 (sd 0,00004).
- src/interpret.py (LightGBM TreeSHAP, 30 ribu baris fold 0) berjalan di unit dice-shap.
