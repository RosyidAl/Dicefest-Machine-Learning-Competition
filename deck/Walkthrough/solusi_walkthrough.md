## First Principles Analysis: Solusi dan Alternatif untuk Studi Kasus Trilema Energi Perkotaan

Dokumen ini memetakan **solusi yang diimplementasikan** dan **alternatif yang tersedia** untuk setiap lapis masalah, berdasarkan kebenaran dasar dari analisis sebelumnya (`first_principles_analysis.md`).

---

### 1. Problem Essence (Pengingat)

**Masalah inti:** Menerjemahkan data anonim 613 ribu kawasan permukiman menjadi (a) prediksi lima besaran energi yang akurat per kawasan, dan (b) peta keputusan per zona yang bisa ditindaklanjuti pemerintah kota dalam menghadapi trilema emisi-keandalan-keterjangkauan.

**Kriteria sukses:**
- Column-Wise NRMSE serendah mungkin (tiket masuk finalis).
- Penjelasan variabel berpengaruh yang didukung data (35% bobot juri).
- Rekomendasi kebijakan per zona yang tahan terhadap perubahan asumsi (bagian dari 35%).

---

### 2. Solusi yang Diimplementasikan (Tim BISMILLAH)

#### 2.1 Lapis Data: Membongkar Struktur Sebelum Memodelkan

| Temuan | Implikasi | Solusi |
|--------|-----------|--------|
| Gas = EUI x luas lantai (selisih 7e-16) | Cukup modelkan EUI, gas mengikuti | Satu pipeline gas/EUI, bukan dua pipeline terpisah |
| Porsi bangunan bersih = round(100 x bld_dim_02 / bld_dim_01, 2) | Tidak perlu model ML | Rumus deterministik |
| Adopsi konstan per subzona, 92% data uji di subzona yang dikenal | Salinan cukup untuk mayoritas | Salinan + 2 LightGBM (cross-entropy + regresi) untuk 8% subzona baru |
| Split test = acak 30% per baris (AUC adversarial 0,4997) | Rata-rata target wilayah sah sebagai fitur | CV acak per baris (KFold), bukan GroupKFold |

**Hasil:** dua dari lima target dijawab tanpa ML. Sisa tiga target (gas/EUI, listrik, adopsi subzona baru) yang benar-benar dimodelkan.

#### 2.2 Lapis Prediksi: Pipeline Tiga Jalur

**Jalur 1: Gas dan EUI (28 model)**
- Target transform: EUI langsung, log EUI, gas per meter, gas per luas, gas per EPC, Poisson.
- Keluarga: XGBoost GPU, LightGBM, MLP (polos, PLR, TabM-mini), MLP multi-task, NCA.
- Fitur kunci: `area_per_meter`, rata-rata target wilayah OOF (subzona, zona, petak iklim, jendela sub_n), kawasan termirip kNN-target OOF, rasio fisik (`gea_ratio`, `area_per_epc`).
- Model tunggal terbaik untuk EUI: xp_gpm (XGBoost Poisson pada gas per meter), NRMSE 0,6574.

**Jalur 2: Listrik (13 model)**
- Target transform: listrik per meter, log listrik per meter, listrik per luas, Poisson.
- Keluarga: XGBoost GPU, LightGBM, MLP, CatBoost (dibuang dari final karena tidak deterministik).
- Model tunggal terbaik: p3_epm_wM (MLP PLR), NRMSE 0,3184.

**Jalur 3: Identitas dan adopsi**
- Porsi bangunan bersih: rumus.
- Adopsi: salinan subzona dikenal + dua model LightGBM untuk subzona baru.

**Ensemble: bobot adaptif (A2xM2xp2)**
- NNLS dengan bobot yang berubah menurut peringkat luas (A), jumlah meter (M), dan besar prediksi (p).
- 2 knot per variabel, total 8 kolom basis x jumlah anggota.
- Gain dibanding NNLS global (D012): skor total -0,000124; EUI 0,65029 menjadi 0,64984.

**Post-processing:**
1. Koreksi galat wilayah (kNN residual per zona, petak iklim, jendela sub_n); besar koreksi (rms/sigma): gas 0,0205, EUI 0,0434, listrik 0,0120.
2. S1: koreksi kawasan bermeter 2-4 (selalu diprediksi terlalu rendah).
3. S3: tarik prediksi gas mendekati batas 1 juta kWh ke bawah (truncated normal).
4. S4: clip ke rentang data latih.
5. Salinan 77 kawasan duplikat.

**Skor akhir:**
- CV (ENS0018, v16 awal): NRMSE 0,2496.
- LB publik: 0,24913.
- LB private: 0,24844 (peringkat 8).

#### 2.3 Lapis Kebijakan: Peta Trilema 330 Zona

- Model penjelas: satu LightGBM per target, hanya fitur kawasan terukur. R2: gas per meter 0,75, EUI 0,56, listrik per meter 0,49, adopsi 0,99.
- SHAP (TreeSHAP, 20 ribu baris sampel): mengidentifikasi pendorong per target.
- Dua ukuran per zona: potensi *retrofit* (TWh) dan kenaikan konsumsi listrik (%).
- Empat kuadran: Q1 (*retrofit* dulu, 118 zona, 47% gas), Q2 (kandidat *heat pump*), Q3 (penguatan gardu dulu), Q4 (program standar).
- Uji sensitivitas: porsi pemanasan 70/80/90%, COP 2,5/3/3,5. Urutan zona tidak berubah.

---

### 3. Alternatif Solusi yang Bisa Diimplementasikan

Berdasarkan COVERAGE.md, ada pendekatan yang statusnya masih QUEUED dan pendekatan lain yang belum tercatat. Berikut analisis per lapis.

#### 3.1 Alternatif Feature Engineering

| Pendekatan | Status | Potensi Gain | Alasan |
|------------|--------|-------------|--------|
| **FB03: Komposisi (entropi, HHI, dominan, indeks era)** | QUEUED | Kecil-sedang | Entropi bauran pemanas menangkap diversifikasi sumber panas; HHI era bangunan menangkap heterogenitas kawasan. Bisa menambah informasi yang tidak ada di rata-rata atau rasio sederhana. |
| **FB08: PCA/SVD transduktif** pada 127 kolom socio_demog | QUEUED | Kecil | Mengurangi dimensi blok terbesar. GBDT sendiri menangani kolinearitas, tapi bisa membantu MLP dan linear. |
| **FB09: KMeans cluster + TE** | QUEUED | Kecil | Arketipe kawasan sebagai fitur kategoris. Mirip dengan apa yang kNN-target sudah tangkap, tapi lebih hemat komputasi. |
| **FB10: Pencarian interaksi** | QUEUED | Tidak pasti | Interaksi eksplisit (era x iklim, ukuran x bauran pemanas) bisa membantu model linear dan MLP, tapi GBDT sudah menangkap ini secara implisit. |
| **FB12: Seleksi fitur (null importance)** | QUEUED | Kecil | Membuang fitur yang tidak pernah lebih penting dari permutasi acak. Mengurangi noise, sedikit mempercepat training. |

**Rekomendasi:** FB03 (komposisi) paling menjanjikan karena menangkap informasi kualitatif yang belum ada. FB08-FB12 kemungkinan memberikan perbaikan marginal di bawah threshold penerimaan 0,2% relatif.

#### 3.2 Alternatif Model

| Pendekatan | Status | Potensi Gain | Alasan |
|------------|--------|-------------|--------|
| **M04: LightGBM GOSS** | QUEUED | Kecil | Sampling berbasis gradien; lebih cepat tapi biasanya tidak lebih akurat dari gbdt. |
| **M08: XGBoost max_bin 1024** | QUEUED | Kecil | Resolusi split lebih halus; berguna untuk target dengan distribusi ekor panjang (gas total). |
| **M15: RealMLP (pytabkit)** | QUEUED | Sedang | Default tuning otomatis yang sering menang di benchmark tabular. Menambah diversitas jika pola galatnya berbeda dari MLP manual. |
| **M20: KNN regressor** | QUEUED | Kecil | Sudah ada kNN-target sebagai fitur; KNN sebagai model utuh kemungkinan redundan. |
| **M21: ExtraTrees** | QUEUED | Kecil | Split acak; biasanya kalah dari boosting di data besar, tapi bisa menambah diversitas ensemble. |
| **M22: Ridge fitur kaya** | QUEUED | Kecil | Baseline linear yang kuat jika feature engineering sudah bagus; bisa menangkap relasi yang GBDT lewatkan. |
| **M24: FT-Transformer** | QUEUED | Sedang | Arsitektur transformer untuk data tabular; kompetitif di beberapa benchmark, tapi lambat dan perlu tuning. |
| **M26: TabPFN/TabICL (foundation tabular)** | QUEUED | Tidak pasti | Model tabular pra-latih; cepat di data kecil, tapi 429 ribu baris mungkin di luar kapasitas. |
| **M27: H2O/FLAML** | QUEUED | Kecil | AutoML framework; berguna untuk menemukan konfigurasi yang terlewat, tapi pipeline sudah cukup matang. |

**Rekomendasi:** M15 (RealMLP) paling menjanjikan untuk diversitas ensemble. M24 (FT-Transformer) layak dicoba jika ada waktu GPU. Sisanya kemungkinan di bawah threshold penerimaan.

#### 3.3 Alternatif Ensemble dan Post-processing

| Pendekatan | Status | Potensi Gain | Alasan |
|------------|--------|-------------|--------|
| **EN2: Rata-rata top-N** | QUEUED | Kecil | Baseline sederhana; biasanya kalah dari NNLS tapi berguna sebagai pembanding. |
| **EN4: Hill climbing (Caruana)** | QUEUED | Kecil | Seleksi greedy dengan pengulangan; kadang menemukan campuran yang NNLS lewatkan. |
| **EN7: Level-2 MLP** | QUEUED | Tidak pasti | Stacking nonlinear; EN6 (Level-2 LGB) sudah gagal, jadi kemungkinan kecil berhasil. |
| **A01: Pseudo-labeling** | QUEUED | Sedang | Menggunakan prediksi test yang percaya diri sebagai label tambahan. Bisa membantu di wilayah yang kurang terwakili. Risiko: overfit ke prediksi sendiri. |
| **A04: Distilasi NN dari ensemble** | QUEUED | Sedang | Satu model MLP yang meniru perilaku 41 model. Lebih cepat di inference, dan sering sedikit lebih baik dari ensemble mentah karena smoothing implisit. |
| **S02: Seed bagging GBDT** | QUEUED | Kecil | Multi-seed untuk XGB/LGB; S01 menunjukkan seed bagging NN memberikan diminishing returns di 6 seed. |

**Rekomendasi:** A01 (pseudo-labeling) dan A04 (distilasi) paling menjanjikan. EN4 layak dicoba karena biayanya rendah.

#### 3.4 Alternatif HPO

| Pendekatan | Status | Potensi Gain | Alasan |
|------------|--------|-------------|--------|
| **H01: Optuna TPE per (target, keluarga)** | QUEUED | Sedang | Tuning manual/grid sudah dilakukan, tapi tuning bayesian sering menemukan kombinasi hiperparameter yang tidak intuitif. |

**Rekomendasi:** Layak dijalankan untuk anggota ensemble yang paling berbobot (XGB EUI wA, LGB EUI wA_tew), karena perbaikan pada anggota berbobot besar terpropagasi ke skor akhir.

#### 3.5 Alternatif Kebijakan (Lapis 3)

| Pendekatan | Deskripsi | Kelebihan | Kekurangan |
|------------|-----------|-----------|------------|
| **Kausal (DiD, IV, RDD)** | Mengestimasi efek intervensi, bukan hanya asosiasi | Rekomendasi lebih kuat; juri mungkin menghargai kehati-hatian kausal | Data observasional tanpa variasi eksogen yang jelas; bisa gagal menemukan instrumen yang valid |
| **Simulasi agen** | Memodelkan perilaku rumah tangga individual | Menangkap efek interaksi (adopsi tetangga mempengaruhi adopsi sendiri) | Butuh asumsi banyak; kompleksitas tinggi; data tidak cukup detail |
| **Optimisasi portofolio zona** | Alokasi anggaran retrofit/heat pump yang memaksimalkan pengurangan emisi per rupiah, dengan batasan kapasitas gardu | Langsung menjawab pertanyaan kebijakan | Butuh data biaya yang tidak ada di dataset |
| **Analisis sensitivitas multidimensi** | Memperluas uji porsi pemanasan dan COP ke dimensi lain (biaya energi, diskon, penyerapan subsidi) | Memperkuat klaim "tahan asumsi" | Bisa terlalu spekulatif tanpa data biaya |

**Rekomendasi:** Pendekatan yang dipakai (SHAP + kuadran + uji sensitivitas) sudah tepat untuk data yang tersedia. Menambahkan klaim kausal tanpa instrumen yang valid akan melemahkan argumen. Optimisasi portofolio layak disebut sebagai arah pengembangan dalam presentasi, tanpa mengklaim sudah dilakukan.

---

### 4. Assumptions Challenged (Khusus Solusi)

| Asumsi | Tantangan | Putusan |
|--------|-----------|---------|
| "Ensemble NNLS global sudah cukup" | Akurasi model berbeda menurut ukuran kawasan; bobot tunggal tidak menangkap ini | **Buang.** Bobot adaptif A2xM2xp2 menurunkan EUI 0,00012 (D012). |
| "CatBoost GPU deterministik" | Rerun memberikan iterasi early-stopping berbeda; selisih prediksi 352 kWh | **Buang.** CatBoost dikeluarkan dari pipeline final (D004). Biayanya: +0,00001 di CV. |
| "Lebih banyak model selalu memperbaiki ensemble" | Seleksi v16 membuang 17 dari 50 anggota tanpa rugi; Poisson (4 anggota) hanya -0,000035 | **Modifikasi.** Diversitas pola galat lebih penting dari jumlah. Anggota yang galatnya berkorelasi tinggi tidak membantu. |
| "Koreksi residual selalu membantu" | Stage-2 LGB pada residual hanya -0,00006; koreksi kNN jauh lebih efektif (gas 0,0205, EUI 0,0434) | **Modifikasi.** Koreksi spasial sederhana (rata-rata tetangga, shrinkage) lebih baik dari model residual yang rumit. |
| "HPO bayesian akan memberikan gain besar" | Tuning manual/grid sudah dekat optimal; gain terbesar datang dari fitur dan ensemble, bukan hiperparameter | **Modifikasi.** HPO layak untuk anggota berbobot besar saja, bukan semua model. |

---

### 5. Ground Truths (Khusus Solusi)

1. **Batas galat dari variabilitas tak teramati.** Kurva a + b/M menunjukkan bahwa kawasan dengan sedikit rumah tangga selalu lebih sulit diprediksi. Tidak ada model atau fitur yang menembus batas ini. Implikasi: perbaikan harus datang dari cara menggabungkan prediksi (ensemble adaptif, koreksi spasial), bukan dari model individual yang lebih besar.

2. **Diversitas galat menggerakkan ensemble.** Perbaikan ensemble datang dari anggota yang galatnya tidak berkorelasi, bukan dari anggota yang sedikit lebih akurat tapi galatnya sama. Contoh dari holdout: MLP PLR p1_lepm lebih lemah sebagai model tunggal (0,32174 vs 0,32035 untuk MLP polos) tetapi memperbaiki blend listrik -0,00056; MLP multi-task m1 adalah anggota EUI tunggal terbaik saat itu (0,66077) tetapi hanya menambah -0,00004 ke blend.

3. **Fitur spasial OOF adalah sumber informasi terbesar setelah fitur fisik.** Rata-rata target wilayah (EUI -0,0022 pada CV acak), kawasan termirip (EUI -0,0022, 5/5 fold), koreksi residual grup (EUI -0,00100 di holdout), dan koreksi residual kNN (tambahan EUI -0,0013 di holdout) memberi perbaikan yang konsisten. Semua dihitung OOF untuk menghindari kebocoran.

4. **Determinisme bukan negosiasi.** Pipeline final harus menghasilkan output identik byte di setiap run. Ini mengeliminasi CatBoost GPU, membatasi multithreading BLAS di tahap assemble, dan memaksa chunk prediksi TabM.

---

### 6. Reasoning Chain: Dari Masalah ke Solusi

```
Masalah                          Kebenaran Dasar                    Solusi Terpilih                  Alternatif
────────────                     ────────────────                   ──────────────                   ──────────
5 target, 613 ribu baris    -->  2 target = rumus/salinan      --> 3 target yang benar-benar     --> Multi-output tunggal
                                                                    dimodelkan                       (M17, M18) -- QUEUED

EUI mendominasi galat       -->  Variabilitas tak teramati      --> Fitur per unit + kNN spasial  --> FB03 (komposisi),
(0,65 dari rata-rata 0,25)       menetapkan batas                   + ensemble adaptif                FB08 (PCA socio)

Akurasi berbeda per         -->  Batas galat ~ a + b/M          --> Bobot ensemble A2xM2xp2       --> EN4 (hill climbing),
ukuran kawasan                                                                                        A04 (distilasi)

Galat berkorelasi           -->  Kawasan mirip memakai           --> Koreksi kNN residual          --> A01 (pseudo-label),
antartetangga                    energi mirip                       (shrinkage n/(n+K))                model residual Stage-2

Juri menilai kebijakan      -->  SHAP = asosiasi, bukan         --> Model penjelas terpisah +     --> Metode kausal (DiD, IV)
(35% bobot)                      kausalitas                         kuadran + uji sensitivitas        -- butuh instrumen valid

Reprodusibilitas wajib      -->  CatBoost GPU tidak              --> Pipeline final: XGB + LGB    --> CatBoost CPU
                                  deterministik                      + PyTorch saja                    (deterministik, tapi lambat)
```

---

### 7. Kesimpulan

**Solusi inti yang diimplementasikan** dibangun dari tiga keputusan kunci:
1. Membongkar struktur data sebelum memodelkan (mereduksi 5 target menjadi 3 masalah ML).
2. Membangun fitur dari fisika energi per unit dan informasi spasial OOF, bukan dari ukuran kawasan mentah.
3. Menggabungkan 41 model dengan bobot adaptif yang berubah menurut karakteristik kawasan.

**Alternatif paling menjanjikan yang belum dicoba:**
- **FB03** (komposisi bauran pemanas/era): menangkap informasi kualitatif.
- **M15** (RealMLP) dan **M24** (FT-Transformer): menambah diversitas ensemble.
- **A01** (pseudo-labeling): memanfaatkan data test.
- **H01** (Optuna) pada anggota berbobot besar: perbaikan yang terpropagasi.

**Wawasan kunci:** Urutan sumber gain (validasi > struktur data > fitur > model > ensemble) tervalidasi secara empiris. Perbaikan terbesar datang dari memahami data (gas = EUI x luas, split acak, fitur spasial OOF), bukan dari memilih algoritma atau tuning hiperparameter.

**Trade-off yang diterima:**
- CatBoost dibuang demi determinisme; biayanya +0,00001 di CV.
- Anggota Poisson hanya memperbaiki 0,000035; diterima karena user meminta hasil maksimal, tapi gain-nya di ambang noise.
- Koreksi spasial memakai lambda dari holdout 30%, bukan dari CV; ini keputusan pragmatis karena fit ulang di nested CV terlalu mahal.
- Rekomendasi kebijakan berbasis asosiasi (SHAP), bukan kausalitas. Cukup untuk konteks kompetisi, tapi perlu peringatan eksplisit.
