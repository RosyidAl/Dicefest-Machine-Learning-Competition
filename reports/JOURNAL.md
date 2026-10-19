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

### [2026-10-17 13:35 WIB] Analisis | kepadatan tetangga koreksi residual (λ di test)
- Hipotesis: λ di-fit dengan kumpulan tetangga P (49% dari total baris), sedangkan di test kumpulannya seluruh train (70%), 1,43x lebih padat. λ mungkin terlalu kecil untuk test.
- Setup (scratchpad density.py, holdout P70/H30, anggota holdout sama dengan xtarget): fitur koreksi dihitung dengan subsampel 70% dari P sebagai kumpulan tetangga, λ di-fit di H (cross-fitted 5-fold), lalu diterapkan ke fitur dari kumpulan P penuh dengan skala 1,0 sampai 1,6.
- Hasil (NRMSE H):

  | Metrik | tanpa koreksi | kumpulan 70% | kumpulan 100% | λ70 di fitur 100% (x1,0) | x1,1-1,2 |
  |---|---|---|---|---|---|
  | gas | 0,24638 | 0,24567 | 0,24538 | 0,24540 | 0,24538 |
  | EUI | 0,65109 | 0,64995 | 0,64959 | 0,64961 | 0,64959 |
  | elec | 0,31503 | 0,31489 | 0,31475 | 0,31479 | 0,31478 |

- Kesimpulan: gain koreksi naik dengan kepadatan (gas -0,00029, EUI -0,00036, elec -0,00014 dari 70% ke 100%), dan λ dari kumpulan yang lebih jarang hampir optimal di kumpulan yang lebih padat (rugi <= 0,00004 per metrik, skala 1,1-1,2 memulihkan <= 0,00001 total). λ tidak diubah. Estimasi holdout untuk gain koreksi di test bersifat konservatif.

### [2026-10-17 13:35 WIB] Fase 12 | SHAP selesai
- src/interpret.py: reports/figures/shap_t3|t2|t5.png + shap_dep_*.png, reports/shap_family.csv, reports/shap_top_features.csv.
- Porsi |SHAP| per keluarga: EUI sosio 25%, rasio fisik 25%, TE spasial 19%, era 11%, dimensi 10%; listrik per meter sosio 35%, rasio fisik 29%, TE 13%; adopsi sisa pangsa pemanas 45%, hvac_dist 27%, sosio 21%.
- Fitur teratas: area_per_meter (EUI turun tajam, listrik per meter naik hampir linear), s002_ratio, area_per_epc, meters_per_epc; adopsi: hvac_rest, hvac_dist_02 (rata-rata 82%, dugaan pangsa gas sentral; < 40% menambah adopsi 5-15 poin).
- Writeup bagian 8 dan slide 8 diisi (tiga dilema kebijakan).

### [2026-10-17 13:38 WIB] E0271-E0278 | V02 CV_SECONDARY (hipotesis)
- Hipotesis: dengan GroupKFold per zona (saran panitia), semua baris validasi berada di wilayah yang tidak terlihat, sehingga TE spasial tidak tersedia dan skor jauh lebih buruk. Di CV_PRIMARY (meniru test, D001) TE memberi gain besar.
  Selisih ini menunjukkan seberapa besar skor bergantung pada informasi wilayah, dan mengapa pemilihan skema CV harus mengikuti mekanisme split.
- Setup: configs/E0271.yaml, src/exp_cv_secondary.py (LightGBM lr 0,1, early stopping di fold validasi, hanya untuk perbandingan relatif).

### [2026-10-17 13:50 WIB] Analisis | bobot koreksi residual bergantung ukuran baris?
- Hipotesis: λ optimal berbeda untuk kawasan kecil dan besar (noise label sebanding 1/M).
- Setup (scratchpad lam_inter.py): fitur koreksi holdout yang sama (xt/own_*.npy), ditambah interaksi dengan log M, log A, dan log M kuadrat; λ cross-fitted 5-fold di H, bootstrap 100x.
- Hasil (delta vs koreksi linear): gas +0,00002 sampai +0,00004, EUI +0,00005 sampai +0,00009, elec -0,00001 sampai +0,00002.
- Kesimpulan: DITOLAK. Koreksi linear sudah optimal; interaksi hanya menambah varians.

### [2026-10-17 13:57 WIB] FB07b | kNN-target diperluas (hipotesis)
- Konteks: user minta fokus memaksimalkan model (notebook belakangan). Koreksi residual masih memberi sekitar -0,0006 total di atas blend v16, artinya model belum menangkap semua sinyal ketetanggaan.
- Hipotesis: menambahkan ke blok FB07 statistik kNN-target di ruang lain dan jendela lain yang terbukti berguna di koreksi residual (ruang era, ruang bangunan + PCA sensus, ruang ukuran 3 fitur, jendela W=10 dan W=1000) memperbaiki model tunggal (target: EUI <= -0,001 dan 4/5 fold).
- Setup: src/exp_fb07b.py, XGB gpm (x4: lr 0,05, depth 8), 5-fold primer, input v16 (agg + TE + sel + FB07). A = FB07, B = FB07 + FB07b (6 konfigurasi, nilai leui/lepm/lgpm + jarak rata-rata). Jalur kode sama untuk A dan B.
- Job lain di /root/kaggle dijeda (SIGSTOP, izin user); cache pip/uv dihapus (izin user).

### [2026-10-17 14:36 WIB] Insiden | run #1 terkena OOM
- dice-run1 dibunuh OOM killer (systemd: Result oom-kill, mem peak 17,2 GB). Penyebab: proses yang dijeda dengan SIGSTOP tetap memegang RAM (jalur run #1 sekitar 11 GB, dua job /root/kaggle sekitar 8 GB, uji CV sekunder 3 GB), ditambah A/B FB07b (2 x 5 GB) dan satu uji kesamaan fitur (sekitar 5 GB).
- Dampak: output 18 anggota yang sudah selesai aman (resume melewati mereka). Progres x8_gpn (fold 6/10) dan p1_gpm (fold 8/10) hilang, sekitar 1,5 jam GPU.
- Pelajaran: SIGSTOP tidak membebaskan RAM. Hitung RAM semua proses yang dijeda sebelum meluncurkan job baru; beri MemoryMax pada setiap unit.
- Uji kesamaan fitur FB07b versi paralel per blok (scratchpad fb07b_eq.py): 5 dari 6 konfigurasi identik; era_w300_k50 berbeda di 0,033% entri (jarak seri di ruang era, maks 0,0046). Deterministik untuk BLOCK tetap.

### [2026-10-17 14:59 WIB] FB07b | hasil A/B (XGB gpm 5-fold)
| Varian | gas | EUI |
|---|---|---|
| A: FB07 | 0,25198 | 0,66243 |
| B: FB07 + FB07b | 0,25167 | 0,66199 |
| Delta | -0,00031 (5/5 fold, SE 0,00015) | -0,00045 (4/5 fold, SE 0,00048) |
- Per fold EUI: +0,00098, -0,00014, -0,00146, -0,00156, -0,00005.
- Kesimpulan: DITOLAK untuk dibangun jadi anggota. Gain sekitar seperlima gain FB07 (EUI -0,0022) dan EUI tidak signifikan. Perkiraan gain blend sekitar -0,00003 setelah koreksi residual, yang sudah memakai ruang tetangga yang sama secara linear. Biaya 9 anggota baru lebih dari 10 jam GPU.
- Catatan teknis: FB07 versi paralel per blok (fb07b.DataP) memangkas waktu satu fold A/B dari sekitar 35 menit menjadi sekitar 2 menit.

### [2026-10-17 15:18 WIB] E0271-E0278 | hasil V02 CV_SECONDARY
| Skema | TE | t1 gas | t3 EUI | t2 elec |
|---|---|---|---|---|
| primer (KFold per baris) | tanpa | 0,25727 | 0,67210 | 0,32623 |
| primer | dengan | 0,25617 | 0,66990 | 0,32546 |
| sekunder (GroupKFold zona) | tanpa | 0,27081 | 0,69231 | 0,33018 |
| sekunder | dengan | 0,36375 | 0,83305 | 0,34736 |
- Kesimpulan: DITERIMA. Di skema yang meniru test, TE memperbaiki gas -0,0011, EUI -0,0022, elec -0,0008. Di GroupKFold per zona, wilayah validasi tidak pernah terlihat: skor memburuk bahkan tanpa TE (EUI +0,020), dan model ber-TE runtuh (EUI 0,833) karena fitur TE baris latih informatif sedangkan baris validasi kosong.
- Implikasi: skema CV harus mengikuti mekanisme split (D001). Dengan fold per zona, TE akan salah dibuang. Bahan jawaban juri.

### [2026-10-17 17:33 WIB] v18 | kandidat anggota (arahan user: "pokoknya hasil terbaik")
- Keputusan: kejar semua gain yang nyata walau kecil (di bawah aturan 0,1%), selama terbukti di blend nested dan dikonfirmasi rantai holdout. D007 ditinjau ulang setelah seleksi v18.
- Kandidat (full 10-fold dulu, holdout hanya untuk yang terpilih):
  - FB07b (src/exp_v18_knn2.py, fb07b.DataX: FB07 dari cache v16 + blok tambahan): xk2_gpm, xk2_gpgea, xk2_lgpm, xk2_lepa, lk2_gpm, lk2_lepa, mk2, pk2_gpgea, pk2_lepm;
  - Poisson (src/exp_v17_pois.py): xp_epa, xp8_gpm, lp_gpm, lp_epm (xp_gpm, xp_eui, xp_epm, xp_gpgea sudah ada).
- Jalur: dice-v18g (GPU, XGB), dice-v18c (CPU, LGB, mulai setelah cache blok tambahan split penuh lengkap), NN menyusul saat VRAM cukup. Penjaga RAM: rantai berhenti bila MemAvailable < 3 GB; OOMScoreAdjust 500 agar run #1 terlindungi.

### [2026-10-17 18:12 WIB] Insiden | OOM GPU 17:37-17:40, semua jalur diluncurkan ulang
- 17:33 rantai v18 GPU (xk2_gpm) diluncurkan saat VRAM bebas hanya sekitar 4,9 GB (job /root/kaggle yang dijeda masih memegang 7,4 GB). Pukul 17:37 xk2_gpm gagal (cudaErrorMemoryAllocation) dan pukul 17:40 jalur XGB run #1 ikut OOM, sehingga run #1 berhenti (anggota yang sudah selesai aman, 25 output).
- 17:40 job /root/kaggle dihentikan atas izin user (VRAM terpakai turun ke 3 GB). mk2 sempat berjalan 3 fold lalu ikut terhenti saat unit dice-v18n di-restart (kesalahan urutan perintah, sekitar 30 menit hilang).
- 18:11 diluncurkan ulang: dice-run1 (resume), dice-v18g (XGB FB07b + Poisson), dice-v18n (NN FB07b), dice-v18c (LGB; mulai bila cache blok tambahan lengkap dan MemAvailable > 10 GB).
- Pelajaran: cek VRAM bebas (bukan hanya RAM) sebelum meluncurkan job GPU; jangan stop unit yang sedang menjalankan job berharga tanpa cek isinya.

### [2026-10-17 19:45 WIB] v18 | anggota pertama + seleksi sementara
- xk2_gpm (full 10-fold, FB07 + FB07b): EUI 0,65928, gas 0,25014 vs xk_gpm 0,65977 / 0,25042 (konsisten dengan A/B E0280).
- Seleksi nested (exp/analysis/select_v18.py, salinan exp/analysis/select_v18_r1.json): v16 0,242467 -> 0,242394 (-0,000073).
  - Gas+EUI 0,896258 -> 0,895944: +xk2_gpm (-0,000213), +xp_gpm, +xp_gpgea, +xp_eui; x8_gpm dan lk_gpm terbuang.
  - Listrik 0,316075 -> 0,316024: +xp_epm; x4_elec, xgb_epm_wM_tew_k10 terbuang.
- Berikutnya: xk2_gpgea, xk2_lgpm, xk2_lepa, xp_epa, xp8_gpm (GPU), lp_gpm, lp_epm, lk2_gpm, lk2_lepa (CPU), mk2, pk2_gpgea, pk2_lepm (NN), lalu seleksi ulang, versi holdout anggota terpilih, fit ulang λ.

### [2026-10-17 22:58 WIB] Fase 11 | run #1 selesai, sha256 beda di bit terakhir
- run #1 (final/run_all.py, resume) selesai 21:44: sha256 53780c9e...e686, sedangkan sub_v16dev e9375ba6...9a7f.
- Diagnosis: 35 array anggota (oof + pred, termasuk adopt_v2) identik byte dengan array pengembangan; skor CV blend identik sampai digit terakhir. Yang berbeda hanya 24 nilai dari sekitar 920 ribu (6 gas, 7 listrik, 11 EUI), selisih relatif maksimum 4e-16.
- Penyebab: OpenBLAS gemv multithread menjumlahkan dengan urutan yang bergantung jumlah thread. Uji: P @ w dengan 8 dan 16 thread berbeda di 5 dan 12 baris dibanding 1 thread; 1 thread selalu identik.
- Perbaikan (D008): tahap assemble dan pekerja koreksi residual di final/src/pipeline.py memakai threadpoolctl.threadpool_limits(1); threadpoolctl==3.7.0 dipin. Uji: assemble run #1 dua kali dengan OMP/OpenBLAS 1 dan 16 thread (scratchpad asm/run.sh), sha256 harus sama.
- Konsekuensi: submisi final harus file keluaran pipeline yang sudah diperbaiki (skornya identik sampai 1e-16 dengan sub_v16dev).
- Hasil uji D008 (23:21): assemble dengan 1 thread dan 16 thread memberi sha256 yang sama, 3767ba10...59be. Disimpan sebagai submissions/sub_v16fin.csv (lolos validate_submission). Ini output pipeline final v16 yang reproducible; skornya sama dengan sub_v16dev.

### [2026-10-17 23:30 WIB] v18 | seleksi sementara (12 dari 13 kandidat full)
- Skor model tunggal (full 10-fold): xk2_gpm EUI 0,65928 / gas 0,25014; xk2_gpgea 0,66074 / 0,25046; xk2_lgpm 0,66229 / 0,25060; lk2_gpm 0,65954 / 0,25046; mk2 0,65851 / 0,24883 / elec 0,31859 (mk1: 0,65934 / 0,24901 / 0,31873); pk2_gpgea 0,65836 / 0,24851; lp_gpm 0,65697 / 0,24881 (terbaik); lp_epm elec 0,31974; lk2_lepa elec 0,32229.
- Seleksi nested (exp/analysis/select_v18_r2.json): total 0,242467 -> 0,242284 (-0,000183). Gas+EUI 0,896258 -> 0,895445, listrik 0,316075 -> 0,315975.
  - Gas ditambah: xk2_gpm, xk2_gpgea, xk2_lgpm, lk2_gpm, mk2_gpm, pk2_gpgea, xp_gpm, xp_eui, xp_gpgea; dibuang: xk_gpgea, xk_lgpm, lk_gpm, mk1_gpm, pk_gpgea, xa_gpgea.
  - Listrik ditambah: mk2_epm, xk2_lepa, xp_epm; dibuang: mk1_epm, x4_elec, xgb_lepm.
  - Versi FB07b menggantikan pendahulunya; lp_gpm (model tunggal terbaik) tidak terpilih.
- Masih berjalan: pk2_lepm, xp8_gpm (full) dan semua versi holdout (selesai sekitar 04:00-05:00). Keputusan v18 menunggu rantai holdout.

### [2026-10-18 02:41 WIB] v18 | seleksi final + rantai holdout
- Semua anggota full dan holdout selesai 02:25 (rc=0). Skor tunggal baru: pk2_lepm elec 0,31884, xp8_gpm EUI 0,65717 / gas 0,24917, xp_epa elec 0,32022, xk2_lepa elec 0,32241.
- Seleksi nested (exp/analysis/select_v18_final.log/.json, 18 kandidat): total 0,242467 -> 0,242265 (-0,000202). Gas+EUI 0,896258 -> 0,895396 (26 anggota), listrik 0,316075 -> 0,315927 (12 anggota).
  - Gas ditambah: mk2_gpm, xk2_lgpm, pk2_gpgea, lp_gpm, xk2_gpgea, xk2_gpm, lk2_gpm, xp_gpgea, xp_eui, xp8_gpm, xp_gpm; dibuang: xa_gpgea, x8_gpm, xk_gpm, mk1_gpm, xk_lgpm, pk_gpgea, lk_gpm.
  - Listrik ditambah: lp_epm, mk2_epm, pk2_lepm, xp_epa, xp_epm; dibuang: xgb_lepm, x4_elec, xgb_epm_wM_tew_k10, pk_lepm, lk_lepa.
- Rantai holdout (src/resid_knn.py fit, semua anggota holdout, exp/resid_knn_fit_v18.log, λ -> exp/resid_knn_v18.json):

  | Metrik | H v16 | H v18 | rantai v16 | rantai v18 |
  |---|---|---|---|---|
  | gas | 0,24643 | 0,24625 | 0,24539 | 0,24533 |
  | EUI | 0,65115 | 0,65071 | 0,64961 | 0,64937 |
  | elec | 0,31508 | 0,31500 | 0,31477 | 0,31476 |

- Kesimpulan: v18 lebih baik di ketiga metrik; gain setelah koreksi sekitar -0,00006 total (sebelum koreksi -0,00014 di H, -0,0002 di OOF nested). Diadopsi (D009, arahan user "hasil terbaik"). Ekspektasi public LB sekitar 0,2493-0,2494.
- Perakitan sub_v18dev: scripts/assemble_dev.py exp/v18cfg (blend v18 + λ v18) -> submissions/sub_v18dev.csv.

### [2026-10-18 02:53 WIB] v18 | sub_v18dev dirakit, pipeline final diperbarui
- sub_v18dev (scripts/assemble_dev.py, exp/v18cfg = blend v18 + λ v18, kode D008): sha256 07f67f8c...135b, lolos validasi. CV blend nested gas 0,245017 / EUI 0,650379 / elec 0,315927, rata-rata 5 target 0,24940 (v16 0,24960). Koreksi rms/sigma lebih kecil dari v16 (gas 0,0210 vs 0,0224, EUI 0,0410 vs 0,0440, elec 0,0129 vs 0,0143): model menangkap lebih banyak sinyal ketetanggaan.
- Selisih dengan v16 (rms / sigma): gas 0,0065, elec 0,0070, EUI 0,0162; korelasi >= 0,9997. BEST_CURRENT = sub_v18dev.
- Pipeline final v18:
  - final/src/fb07b.py (salinan src/fb07b.py dengan sakelar use_x);
  - pipeline.make_data: satu DataX per jalur, FB07b aktif per anggota (knn_x: true), dipakai juga oleh inference;
  - members.yaml: 39 anggota (11 anggota v16 yang tidak terpakai dibuang, 16 baru); resid_knn.json = λ v18; salinan v16 di members_v16.yaml / resid_knn_v16.json (Final B).
- Bug ditemukan di final/make_notebooks.py: features_knn.py tidak ikut tertanam (notebook v16 akan gagal di anggota kNN pertama). Diperbaiki, fb07b.py juga ditambahkan.
- 02:53 run final v18 dilanjutkan di final/work/run1 (anggota v16 dipakai ulang, 16 anggota baru dilatih) -> final/output/submission_run1_v18.csv. Lalu uji drift: array baru harus identik dengan array pengembangan, sha256 harus sama dengan sub_v18dev.

### [2026-10-18 04:05 WIB] Insiden | RAM kritis di run final v18, jalur CPU dihentikan
- Run final v18 (dice-run1) memakai 26,6 GB (puncak 29,4 GB, swap 1 GB): ketiga jalur (xk2_gpm, pk2_gpgea, lk2_gpm) menghitung blok FB07b untuk split yang sama secara bersamaan, masing-masing dengan 8 thread, ditambah DataX + ruang kemiripan per jalur. Claude Code menghentikan shell pemantau karena memori rendah.
- Tindakan: jalur CPU (pid 275317, lk2_gpm fold 4/10) dihentikan manual agar OOM killer tidak menghentikan seluruh unit (OOMPolicy stop). RAM turun ke 14 GB terpakai. Jalur XGB dan NN berlanjut.
- Tindak lanjut: setelah jalur XGB dan NN selesai, run_all di-resume untuk lk2_gpm, lp_gpm, lp_epm lalu adopsi + assemble.
- Pelajaran untuk notebook: jalur paralel dengan blok FB07b butuh RAM lebih besar. Opsi: kurangi THREADS di fb07b atau jalankan jalur CPU setelah cache FB07b terisi.

### [2026-10-18 06:40 WIB] LB sub_v18dev 0,24936 + v19 diluncurkan
- Public LB sub_v18dev: 0,24936 (v16 0,24943, -0,00007). Estimasi rantai holdout -0,00006: validasi tetap akurat.
- User minta lebih maksimal lagi. Hipotesis v19: dua sumber gain v18 (FB07b dan objektif Poisson) belum pernah digabung; anggota Poisson v18 memakai input v16. Poisson + FB07b diharapkan memberi model tunggal terbaik baru dan keragaman blend.
- Kandidat (src/exp_v18_knn2.py, input FB07b): lp2_gpm, lp2_epm (LGB Poisson), xp2_gpm, xp2_eui, xp2_gpgea, xp2_epm, xp2_epa, xp82_gpm (XGB Poisson), pk2_lgpm (PLR, pengganti pk_lgpm yang masih di blend v18).
- Jalur: dice-v19c (CPU), dice-v19g (GPU, mulai bila MemAvailable > 12 GB), dice-v19n (NN, setelah jalur NN run final v18 selesai). Status: logs/v19_guard.log. Full lalu holdout untuk setiap kandidat.
- Run final v18 berjalan di latar (jalur XGB di xp_eui, jalur NN selesai mk2 dengan skor identik dengan dev).

### [2026-10-18 06:50 WIB] v20 | kandidat: versi kNN + FB07b dari anggota lama tanpa kNN
- Dasar: di v18 sebagian besar gain datang dari versi FB07b yang menggantikan pendahulunya. Blend v18 masih memuat 18 anggota tanpa input kNN.
- Kandidat (src/exp_v18_knn2.py, formulasi/bobot/parameter sama dengan anggota lama, input + agg + TE + FB07 + FB07b):
  - XGB: xk2_gpn (x8_gpn), xk2_eui_wsM (xgb_eui_wsM), xk2_gas (xgb_gas), xk2_leui_wA (xgb_leui_wA), xk2_elec (xgb_elec);
  - LGB: lk2_gas (la_gas), lk2_leui_wA (la_leui_wA), lk2_epa (l4_epa);
  - NN: nk2_gpm (n4f_gpm, 60/37 epoch), nk2_lgpm (n6_lgpm, 40/39), nk2b_lepm (n6b_lepm, 24/23), pk2_gpm (p1_gpm, epoch PLR-kNN 12/11), tk2_lepm (t1_lepm, TabM).
- Jalur dice-v20c / dice-v20g / dice-v20n (status logs/v20_guard.log), masing-masing menunggu jalur v19 / run final yang setara selesai dan MemAvailable > 12 GB.
- Rencana: seleksi gabungan v19 + v20, fit ulang λ, sub_v20dev besok pagi; pipeline final + notebook setelahnya (notebook paling lambat mulai 20 Okt siang).

### [2026-10-18 09:03 WIB] v19 berjalan, v20 dibatalkan (permintaan user)
- User: setelah v19 selesai, server dipakai untuk proyek kaggle-nya. Jalur v20 (dice-v20c/g/n) dibatalkan; xk2_gpn full dihentikan di tengah jalan.
- Skor v19 sejauh ini (full 10-fold): lp2_gpm EUI 0,65660 / gas 0,24881 (model tunggal terbaik); xp2_gpm 0,65684 / 0,24889 (xp_gpm 0,65741 / 0,24899); xp2_eui 0,65882 / 0,24860; xp2_gpgea 0,65856 / 0,24891 (xp_gpgea 0,65985 / 0,24911); xp2_epm elec 0,31973; xp2_epa elec 0,32001.
- Run final v18 berakhir 08:31 (jalur XGB dan NN rc=0, jalur CPU dihentikan manual): 48 output; lk2_gpm, lp_gpm, lp_epm menyusul di run final berikutnya.
- Sisa jadwal DICE FEST setelah v19: run final anggota baru (sekitar 8 jam) + notebook headless (sekitar 30 jam). Server perlu kembali paling lambat Selasa 20 Okt malam.

### [2026-10-18 10:45 WIB] v19 | skor anggota + seleksi
- Full 10-fold: xp82_gpm EUI 0,65638 / gas 0,24888 (model tunggal terbaik), lp2_gpm 0,65660 / 0,24881, xp2_gpm 0,65684 / 0,24889, xp2_gpgea 0,65856 / 0,24891, xp2_eui 0,65882 / 0,24860, pk2_lgpm 0,66193 / 0,24888; elec lp2_epm 0,31967, xp2_epm 0,31973, xp2_epa 0,32001.
- Seleksi nested di atas blend v18 (exp/analysis/select_v19.log/.json): 0,242265 -> 0,242231 (-0,000034).
  - Gas+EUI 0,895396 -> 0,895264: +xp82_gpm, +xp2_gpgea, +lp2_gpm, +pk2_lgpm, +xp2_eui; dibuang la_leui_wA, pk_lgpm, xk2_lgpm, xk2_gpm.
  - Listrik 0,315927 -> 0,315894: +lp2_epm, +xp2_epa; dibuang xk_lepa, xp_epm.
- Gain sekitar seperenam gain v18: model tunggal lebih baik tetapi informasinya tumpang tindih dengan anggota v18. Keputusan menunggu rantai holdout (4 holdout XGB terakhir sedang berjalan).

### [2026-10-18 11:42 WIB] v19 diadopsi, sub_v19dev dirakit, server diserahkan ke user
- Rantai holdout (exp/resid_knn_fit_v19.log, λ -> exp/resid_knn_v19.json) vs v18: gas 0,24533 -> 0,24531, EUI 0,64937 -> 0,64930, elec 0,31476 -> 0,31475 (total sekitar -0,00002). Seleksi nested -0,000034.
- Keputusan D010: v19 = kandidat Final A (arahan user: maksimal); semua metrik membaik walau tipis.
- sub_v19dev (exp/v19cfg = blend v19 + λ v19): sha256 62444504...f216, lolos validasi. CV blend nested gas 0,244976 / EUI 0,650288 / elec 0,315894, rata-rata 5 target 0,24937. BEST_CURRENT = sub_v19dev.
- Konfigurasi final diperbarui ke v19 (final/configs/members.yaml: 40 anggota; resid_knn.json = λ v19); salinan v18 (members_v18.yaml, resid_knn_v18.json) dan v16 disimpan.
- final/src/fb07b.py: THREADS 8 -> 3 untuk notebook (nilai tidak bergantung jumlah thread, hanya RAM), mencegah lonjakan RAM saat tiga jalur menghitung FB07b bersamaan.
- Semua job DICE FEST berhenti 11:45; server dipakai user untuk proyek kaggle. Sisa: run final anggota baru (lk2_gpm, lp_gpm, lp_epm dari v18 + 7 anggota v19) sekitar 6-8 jam, lalu notebook headless sekitar 30 jam. Server perlu kembali paling lambat Selasa 20 Okt malam.

### [2026-10-18 18:35 WIB] LB sub_v19dev 0,24934; jeda untuk migrasi akun
- Public LB sub_v19dev 0,24934 (prediksi rantai holdout sekitar 0,24934). Riwayat: v16 0,24943, v18 0,24936, v19 0,24934.
- User sedang memakai server untuk proyek /root/kaggle dan akan memigrasi akun Claude: tidak ada proses DICE FEST yang dijalankan. Langkah lanjutan lengkap di reports/STATUS.md bagian HANDOFF.

### [2026-10-19 13:01 WIB] Lanjut setelah migrasi akun (Claude Code kini akun rayhanmulachela@ada4career.com)
- Uji drift anggota v18 di final/work/run1 (dilatih pipeline final 17-18 Okt): 13 anggota (xk2_gpm, xk2_lgpm, xk2_gpgea, pk2_gpgea, pk2_lepm, mk2_gpm, mk2_epm, xp_gpm, xp_eui, xp_epm, xp_gpgea, xp_epa, xp8_gpm) identik byte dengan exp/oof|pred.
- GPU dipakai proses di luar container (PID host 931477, 13,2 GB VRAM); tidak terlihat dan tidak bisa dihentikan dari sini. User diminta mematikannya dari sisi lain.
- 13:00 jalur CPU run final v19 dijalankan sendiri (dice-run1cpu: lk2_gpm, lp_gpm, lp_epm, lp2_gpm, lp2_epm). Jalur XGB (xp82_gpm, xp2_gpgea, xp2_eui, xp2_epa) dan NN (pk2_lgpm) menunggu GPU.
- 13:0x atas permintaan user semua job DICE FEST dihentikan (dice-run1cpu dihentikan saat lk2_gpm fold awal; belum ada anggota baru yang selesai). Tidak ada unit dice-* yang aktif. Langkah lanjutan tetap sesuai HANDOVER.md bagian 7.

### [2026-10-19 16:30 WIB] Run final v19 dilanjutkan, berbagi server dengan proyek kaggle
- User: lanjutkan DICE FEST sekarang; server juga dipakai proyek kaggle (job train_noisy_student.py).
- Resource 16:16: RAM bebas 28 GB (job kaggle 5,5 GB, sekitar 1 core); VRAM terpakai 15,5/16 GB = proses luar container (PID host 931477, 13,2 GB, ada sejak siang) + job kaggle (2,3 GB); disk bebas 11 GB.
- dice-run1s (logs/run1_shared.sh): jalur CPU langsung jalan (OMP 12 seperti pipeline.LANE_THREADS, Nice 5 supaya job kaggle didahulukan); jalur XGB + NN menunggu VRAM bebas >= 10 GB agar job kaggle tetap punya ruang; semua jalur dihentikan bila MemAvailable < 2,5 GB; setelah ketiga jalur rc=0 assemble -> final/output/submission_run1_v19.csv dan sha256 dibandingkan dengan sub_v19dev.
- Perkiraan dari waktu dev: jalur CPU sekitar 5,4 jam (lk2_gpm 57 menit, lp_gpm 74, lp_epm 44, lp2_gpm 92, lp2_epm 55); jalur XGB sekitar 2,6 jam + NN setelah VRAM tersedia. Disk: 10 anggota baru sekitar 5-6 GB (cukup); notebook penuh butuh sekitar 23 GB lagi.

### [2026-10-19 16:35 WIB] Arahan baru user + PP0005 adopsi v3 (hipotesis)
- User: notebook ditunda; fokus submisi terbaik; boleh submit langsung ke Kaggle (token diberikan, disimpan di file privat di luar repo); target user peringkat 3/4 di LB publik.
- LB publik 19 Okt 16:40: #3 Il Principe 0,23857, #4 Hbat 0,24809, #5 opo yo 0,24897, #6 BISMILLAH 0,24934, #7 We are him 0,24936, #8 nga tau 0,24940. Kuota hari ini 5 (belum dipakai). Penilaian jujur ke user: peringkat 4 butuh -0,00126 (v16 -> v19 total -0,00009); celah sah tersisa sekitar 0,0001 sampai 0,0004.
- Adopsi untuk subzona baru adalah tuas sah terbesar yang tersisa: NRMSE test sekitar 0,041 (8,1% baris test, RMSE unseen-weighted 1,030 vs lantai binomial sekitar 0,944), batas gain sekitar -0,0006 total.
- PP0005 (src/exp_adopt_v3.py, CV 5-fold per subzona sama dengan adopt_v2): (a) cknn = pooled count_r/K dari k subzona berlabel paling mirip di ruang PCA sensus (k 10/30/100, out-of-fold); (b) wmix = bobot latih yang menyamakan campuran nrow berlabel dengan campuran subzona test baru; (c) gabungan. Ukuran: RMSE unseen-weighted, baseline 1,0303. Kriteria: < 1,025 baru layak dipakai.

### [2026-10-19 16:46 WIB] v21 (hipotesis): LightGBM Poisson + FB07b untuk formulasi xp2_* (CPU, GPU masih terpakai)
- lp2_gpm/lp2_epm (LGB Poisson + FB07b) adalah model tunggal terbaik di v19 dan masuk blend bersama versi XGB-nya. Pasangan LGB untuk formulasi xp2_* yang terpilih belum ada: lp2_eui (gas per luas, bobot iA), lp2_gpgea (gas per bld_dim_03, iA), lp2_epa (listrik per luas, iA).
- Rencana: full 10-fold + hold P70/H30 (src/exp_v18_knn2.py, rantai logs/dev_chain.sh, guard RAM 3 GB), lalu seleksi nested di atas v19 + rantai holdout. Perkiraan gain kecil (-0,00002 sampai -0,00004).

### [2026-10-19 17:55 WIB] PP0005 hasil: adopsi v3 ditolak (D011); submisi pertama hari ini
- CV unseen-weighted (logs/pp0005_adopt3.log): base 1,0303 (sama dengan adopt_v2, byte identik lewat src/adopt_make.py), cknn 1,0297, wmix 1,0346, cknn_wmix 1,0335, epc 1,0307. Rata-rata 8 model (src/adopt_blend.py) 1,0283 (NNLS jujur 1,0284). Efek ke skor total hanya sekitar -0,000016.
- Submit sub_v19a3 (v19 + adopt_v3a, sha 9e745796...) 17:44: LB publik 0,24938, lebih buruk 0,00004 dari sub_v19dev. Simulasi subset publik dari galat CV: -0,000016 +- 0,000017, peluang >= +0,00003 sekitar 0,3%. Diperlakukan sebagai perbedaan sistematis; adopt_v2 dipertahankan. Sisa varian (cknn_epc, hp1, hp2) dihentikan.
- Pelajaran: CV per subzona berlabel bukan proksi sempurna untuk subzona test baru; perubahan adopsi berikutnya butuh bukti LB.
- Kuota hari ini: 4 tersisa.

### [2026-10-19 18:05 WIB] ENS0019 (hipotesis): bobot blend yang berubah mulus dengan ukuran bangunan
- Uji cepat di blend v19 (skor nested 5-fold jujur, scratchpad blend_bins.py / blend_vc.py): NNLS per tertil luas lantai (bld_dim_06) total 0,242197 vs global 0,242231 (-0,000034); bobot linear sepotong-sepotong pada peringkat luas (2 knot) 0,242192 (-0,000039), 3 knot 0,242190; per M atau A/M lebih kecil. Gain terbesar di EUI (0,650288 -> 0,650139).
- Hipotesis: anggota berbeda unggul di bangunan kecil vs besar; bobot yang bergantung ukuran (fungsi dari fitur, bukan target) menangkapnya. Implementasi: src/blend.py size_basis / nnls_blend_vc / predict_vc (peringkat = ECDF luas baris fit).
- Uji penentu: rantai holdout dengan koreksi residual (src/chain_vc.py 2 knot, exp/chain_vc2.log) vs v19 (gas 0,24531, EUI 0,64930, elec 0,31475). Diadopsi bila ketiga metrik tidak memburuk dan total membaik.
- Run final v19: jalur CPU dijeda (SIGSTOP) 17:49 karena dua proses LightGBM 12 thread saling memperlambat (lp2_eui fold 0 butuh 73 menit).

### [2026-10-19 18:55 WIB] ENS0019 hasil: blend varying-coefficient, LB 0,24925 (D012)
- Nested (scratchpad blend_vc*.py): A2 -0,000039, AxM -0,000077, p -0,000067, A x p -0,000087, A2xM2xp2 -0,000124 (terbaik; A3xM2xp2 0,242111, A2xM2xp3 0,242104).
- Rantai holdout A2xM2xp2 (exp/chain_vc_AMp.log): gas 0,24514, EUI 0,64878, elec 0,31462 (v19 0,24531 / 0,64930 / 0,31475). A2: 0,24529 / 0,64909 / 0,31469.
- sub_v19vc (exp/v19vc = blend v19 + exp/resid_knn_vc_AMp.json; sha 99148e8e...b428) submit 18:54: LB publik 0,24925 (v19 0,24934). BEST_CURRENT = sub_v19vc. Kuota hari ini tersisa 3.
- Insiden: 18:41 guard RAM rantai v21 mematikan lp2_eui di fold 4 (dua assemble paralel menekan MemAvailable < 3 GB). Pelajaran: jangan jalankan dua assemble sekaligus saat job dev besar berjalan. Rantai v21 diluncurkan ulang 18:53.

### [2026-10-19 19:30 WIB] ENS0019 lanjutan: variabel pengondisi tambahan
- Nested (NNLS lewat persamaan normal + Cholesky, hasil identik dengan NNLS langsung untuk A2xM2xp2 = 0,242107): + s (subzona terlihat oleh model fold) 0,242123 (lebih buruk); + d (koefisien variasi prediksi anggota) 0,242074 (-0,000033 lagi; total -0,000157 vs v19). s saja 0,242243, d saja 0,242185.
- blend.blend_vc kini menambahkan 'd' (disagreement) selain 'p'. Rantai holdout A2xM2xp2xd2 berjalan (exp/chain_vc_AMpd.log).
- Uji yang memakan RAM besar (A3xM3xp3, 7,8 GB) dihentikan supaya guard RAM rantai v21 tidak terpicu lagi.
- 19:41 rantai holdout A2xM2xp2xd2: gas 0,24528 / EUI 0,64880 / elec 0,31462 vs A2xM2xp2 0,24514 / 0,64878 / 0,31462 -> 'd' ditolak. Dugaan: prediksi H/test adalah rata-rata 10 model fold sehingga sebaran ketidaksepakatannya bergeser dari OOF; nested CV (hanya OOF) tidak bisa melihat ini. Dicek juga A2xM2 tanpa p (exp/chain_vc_AM.log) untuk menilai ketahanan 'p'.
- 19:56 rantai A2xM2 (tanpa p): gas 0,24525 / EUI 0,64892 / elec 0,31465, lebih buruk dari A2xM2xp2 di ketiga metrik -> 'p' bertahan di holdout; spesifikasi tetap A2xM2xp2. Seleksi anggota di bawah blend VC berjalan (exp/analysis/select_vc.py, log exp/analysis/select_vc.log).
- 20:01 seleksi anggota di bawah blend A2xM2xp2 (exp/analysis/select_vc.log): gas+EUI tidak bertambah (kandidat hasil saring mendapat bobot 0); elec +nn5_epa_te, +xgb_epm_wM_ad: 0,315877 -> 0,315844 (total -0,0000066). Tidak dipakai: anggota generasi awal yang belum ada di pipeline final, gain di bawah noise.
- 20:05 uji nested variabel murni fitur di atas A2xM2xp2: u (urban), c (EPC per meter), r (rasio bersih), t (env_climate_01), h (bld_dim_17).
- 20:07 variabel murni fitur di atas A2xM2xp2 (nested): u -0,000002, c -0,000004, r +0,000022, t +0,000012, h -0,000005 -> tidak ada yang berarti; intersep bebas per basis +0,000009 (ditolak). Spesifikasi final tetap A2xM2xp2.
- PP0006 (hipotesis): konfigurasi kNN koreksi residual dipilih untuk blend v15; dengan blend VC dan anggota ber-kNN, pilihan optimal bisa berbeda. Seleksi maju serakah di H (exp/analysis/select_corr.py, grid 39 konfigurasi per metrik, kolom di-cache di exp/corrsel/), dibandingkan dengan CONFIGS sekarang pada skor yang sama.
