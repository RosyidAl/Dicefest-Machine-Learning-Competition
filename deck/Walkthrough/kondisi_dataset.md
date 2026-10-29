# Kondisi Dataset: Struktur, Kualitas, dan Implikasinya bagi Pemodelan

Dokumen ini menguraikan kondisi dataset DICE FEST 2026 Machine Learning sebelum pemodelan dilakukan: ukuran dan
satuan baris, kelompok fitur, hierarki lokasi, kualitas data, sifat kelima target, hubungan data latih dengan data
uji, serta batas akurasi yang ditentukan oleh *noise* (galat acak) pada label. Setiap bagian memuat bukti dan
implikasinya bagi tahap berikutnya. Uraian tahapan pemodelan ada di `walkthrough_tahapan.md`, sedangkan rumusan
masalah ada di `problem_statement.md`.

Sumber angka: `reports/eda/EDA_REPORT.md`, `reports/eda/findings.json` (dibangkitkan `src/eda.py`), dan pemeriksaan
ulang langsung terhadap `data/raw/train.csv` serta `data/raw/test.csv` pada 29 Oktober 2026. Figur pendukung ada di
`reports/eda/figures/`.

## Ringkasan

| Aspek | Kondisi | Implikasi |
|---|---|---|
| Ukuran | 429.303 baris latih, 183.987 baris uji (rasio 70:30 tepat) | data cukup besar untuk GBDT dan MLP; 10 fold tetap memadai |
| Satuan baris | agregat satu kawasan permukiman (setingkat kode pos), bukan satu rumah tangga | target makro berskala ukuran kawasan |
| Fitur | 201 fitur numerik bernama samaran dalam lima kelompok, ditambah dua kode lokasi | arti kolom harus direkonstruksi dari data |
| Kebersihan | NaN hanya di dua kolom (695 baris latih), satu nilai sentinel, 11 kolom konstan | praktis tidak perlu imputasi |
| Target | dua target ditentukan rumus pasti, satu konstan per subzona | masalah ML efektif tersisa tiga |
| Split | acak per baris, tanpa *drift* (AUC *adversarial validation* 0,4997) | KFold acak per baris meniru data uji |
| *Noise* label | sekitar 75% varians residual berasal dari rata-rata antarrumah tangga | skor di bawah sekitar 0,21 tidak dapat dicapai tanpa data eksternal |

## 1. Ukuran dan satuan baris

**Bukti.**
- Data latih berukuran 429.303 baris × 209 kolom; data uji 183.987 baris × 204 kolom. Selisih lima kolom adalah
  kelima target. Total 613.290 baris.
- Porsi data uji tepat 183.987 / 613.290 = 0,300000.
- Satu baris adalah agregat satu kawasan permukiman. Rata-rata luas lantai (`bld_dim_06`) per baris 1.711 m²
  (median 1.449 m², rentang 62,7 sampai 11.456 m²), dan rata-rata jumlah meter (`bld_dim_12`) 22 (median 19,
  rentang 2 sampai 157).
- Dalam memori, data latih menempati sekitar 736 MB.

**Implikasi.** Karena setiap baris merupakan agregat beberapa puluh rumah tangga, target konsumsi gas dan listrik
tumbuh hampir sebanding dengan ukuran kawasan. Hal ini menjadi dasar formulasi target per unit (per meter atau per
m²) pada tahap pemodelan.

## 2. Kelompok fitur

Nama fitur disamarkan oleh panitia, tetapi awalan kolom menunjukkan kelompoknya.

| Awalan | Jumlah kolom | Isi menurut panitia | Tingkat pengamatan |
|---|---|---|---|
| `bld_dim_*` | 43 | dimensi dan geometri bangunan | per baris (kawasan) |
| `bld_era_*` | 15 | sebaran era konstruksi | per baris (kawasan) |
| `hvac_dist_*` | 10 | bauran sistem pemanas | per subzona |
| `env_climate_*` | 6 | beban termal iklim mikro | per subzona/sel grid |
| `socio_demog_*` | 127 | okupansi, kepadatan, aktivitas, kepemilikan | per subzona |
| `zone_region_id`, `subzone_id` | 2 | kode lokasi | hierarki lokasi |
| `id` | 1 | nomor baris | tidak dipakai |

Sebanyak 113 dari 132 kolom sensus (`socio_demog`, `hvac_dist`, `env_climate`) bernilai konstan di dalam satu
subzona. Kolom-kolom tersebut dengan demikian merupakan atribut *Output Area* (OA, unit sensus kecil), sedangkan
kolom `bld_dim` dan `bld_era` merupakan atribut baris.

### 2.1 Kolom yang artinya berhasil direkonstruksi

| Kolom | Arti | Bukti |
|---|---|---|
| `bld_dim_06` | luas lantai (m²) | gas = EUI × `bld_dim_06`, selisih relatif maksimum 6,7e-16 |
| `bld_dim_12` | jumlah meter | rasio listrik/`bld_dim_12` paling stabil (log-std 0,318) |
| `bld_dim_01`, `bld_dim_02` | jumlah sertifikat EPC dan jumlah bangunan bersih | porsi bangunan bersih = round(100 × `bld_dim_02` / `bld_dim_01`, 2) di 100% baris |
| `bld_dim_13` | jumlah bangunan tidak bersih | `bld_dim_13` = `bld_dim_01` − `bld_dim_02` |
| `bld_dim_18` | kategori ordinal (empat tingkat, "High (<0%)" sampai "Very Low (>18%)") | satu-satunya kolom kategorikal bangunan |
| `socio_demog_016` | Urban/Rural | 376.496 Urban, 52.807 Rural di data latih |
| `hvac_dist_*` | porsi rumah tangga per jenis pemanas | jumlahnya tidak 100 (median 96,1); sisanya kategori yang tidak dicantumkan |
| `env_climate_01/03/05` | koordinat sel grid iklim | sekitar 15 baris latih per sel |

Dari `hvac_dist` juga dapat direkonstruksi jumlah rumah tangga N per OA secara tepat, yaitu N terkecil yang
membuat semua `hvac_dist` × N / 100 bernilai bulat (median 128). Angka ini dipakai pada model adopsi pemanas listrik.

### 2.2 Blok komposisi

Beberapa kelompok kolom `socio_demog` berjumlah tepat 100 per baris, misalnya 026 sampai 031, 034 sampai 038, 040
sampai 045, 046 sampai 054, 055 sampai 060, 061 sampai 069, 089 sampai 098, dan 099 sampai 106. Kelompok ini adalah
porsi kategori sensus (status kepemilikan, tipe hunian, dan sejenisnya). Sifat komposisi ini menimbulkan
kolinearitas sempurna di dalam blok, yang tidak merugikan model berbasis pohon.

## 3. Hierarki lokasi

**Bukti.**
- Terdapat 330 zona (`zone_region_id`). Jumlah baris latih per zona berkisar 4 sampai 7.072 (median 1.021,5).
- Terdapat 163.185 subzona (`subzone_id`) gabungan latih dan uji: 153.508 di data latih, 110.548 di data uji,
  100.871 beririsan.
- Subzona berukuran kecil: median 4 baris (latih dan uji digabung), maksimum 18, dan 12,2% subzona hanya berisi
  satu baris.
- Setiap subzona berada di tepat satu zona. Nomor subzona (`sub_n`) berurutan secara spasial: 99,07% pasangan
  subzona bernomor berurutan berada di zona yang sama.
- `env_climate_01/03/05` membentuk sel grid iklim dengan sekitar 15 baris latih per sel. `env_climate_02` dan
  `env_climate_04` adalah grid yang sama dengan pengisi 0,0.

**Implikasi.** Struktur bertingkat ini (zona, sel iklim, jendela `sub_n`, subzona) menyediakan kelompok untuk
*target encoding* (pengodean kategori dengan rata-rata target) dan koreksi residual spasial. Semua statistik
berbasis target pada kelompok tersebut wajib dihitung *out-of-fold* (dari lipatan yang tidak memuat baris yang
diprediksi).

## 4. Kualitas data

| Pemeriksaan | Temuan | Penanganan |
|---|---|---|
| Tipe data | semua numerik kecuali `bld_dim_18` (ordinal, empat kategori) dan `socio_demog_016` (Urban/Rural) | di-*encode* menjadi 0 sampai 3 dan 1/0 |
| Nilai hilang | NaN hanya di `bld_dim_04` dan `bld_dim_07`, pada baris yang sama: 695 baris latih (0,16%) dan 336 baris uji | dibiarkan; GBDT menangani NaN secara bawaan |
| Nilai tak hingga | tidak ada | tidak perlu penanganan |
| Sentinel | satu nilai 9999 di `bld_dim_03` (data latih) | tidak berpengaruh berarti |
| Kolom konstan | 11 kolom: `bld_era_01`, `env_climate_06`, `socio_demog_001`, `004`, `026`, `034`, `039`, `045`, `054`, `060`, `069` | dibuang (`common.constant_cols`) |
| Redundansi | 6 pasangan dengan \|r\| > 0,999 (misalnya `socio_demog_032` = 100 − `socio_demog_033`); 75 pasangan \|Spearman\| > 0,95 | dibiarkan; tidak merugikan GBDT |
| Duplikat persis | tidak ada, baik di latih, uji, maupun lintas latih-uji | tidak perlu penanganan |
| Duplikat berkunci | 77 baris uji identik dengan baris latih pada kunci (`socio_demog_005`, subzona, `bld_dim_12`, `bld_dim_01`, `bld_dim_06`); di data latih, 92 grup berkunci sama selalu bertarget identik | target disalin (`src/dup_fix.py`) |
| Kolom `id` | Spearman `id` terhadap target \|ρ\| ≤ 0,0026; `id` hanya nomor urut setelah split | tidak dipakai sebagai fitur |

Secara umum dataset dalam keadaan bersih. Pekerjaan pembersihan terbatas pada membuang kolom konstan,
meng-*encode* dua kolom teks, dan menyalin target untuk 77 baris duplikat.

## 5. Kondisi target

### 5.1 Distribusi

| Target | σ beku | Min | Median | Maks | *Skewness* | Porsi 1% teratas pada jumlah kuadrat deviasi |
|---|---|---|---|---|---|---|
| Gas (kWh) | 151.750,04 | 4.324 | 193.549 | 999.765 | 1,22 | 14,4% |
| Listrik (kWh) | 41.725,37 | 2.968 | 53.839 | 396.492 | 1,23 | 14,8% |
| EUI gas (kWh/m²) | 40,65 | 5,1 | 134,8 | 648,1 | 1,30 | 20,2% |
| Porsi bangunan bersih (%) | 12,49 | 8,33 | 96,3 | 100 | −1,57 | 14,6% |
| Adopsi pemanas listrik (%) | 7,18 | 0 | 3,10 | 93,2 | 3,40 | 34,9% |

Pengamatan:
- σ beku pada metrik sama dengan simpangan baku data latih (ddof 1).
- Target makro dan EUI menceng ke kanan; sebagian kecil baris besar menyumbang porsi besar galat kuadrat.
- Gas terpotong di bawah 1 juta kWh (maksimum 999.765).
- Nilai nol hanya ada pada adopsi (33.859 baris). Tidak ada pelanggaran rentang 0 sampai 100 pada target persen.
- Korelasi Pearson gas-listrik 0,887; gas-EUI hanya 0,18; pasangan lain di bawah 0,11.

Figur: `reports/eda/figures/target_hist.png`.

### 5.2 Relasi pasti antartarget dan fitur

| Relasi | Hasil | Akibat |
|---|---|---|
| gas = EUI × `bld_dim_06` | selisih relatif maksimum 6,7e-16 | gas dan EUI diprediksi oleh satu keluarga model |
| porsi bersih = round(100 × `bld_dim_02` / `bld_dim_01`, 2) | tepat di 100% baris | dihitung dengan rumus, NRMSE 0 |
| adopsi konstan per subzona | berlaku di semua subzona | disalin dari subzona yang dikenal; model hanya untuk sekitar 8% baris uji |
| adopsi ≤ sisa porsi `hvac_dist` | berlaku di 100% baris, R² linear 0,978 | adopsi dimodelkan sebagai sisa pemanas dikurangi satu kategori yang hilang |

Figur: `reports/eda/figures/physics_gas_eui_area.png`.

### 5.3 Kekuatan sinyal fitur

- Gas dan listrik terutama ditentukan ukuran kawasan: Spearman `bld_dim_06` terhadap gas 0,92, `bld_dim_12`
  terhadap listrik 0,90.
- EUI lemah terhadap fitur tunggal; Spearman tertinggi hanya 0,18 (`bld_dim_32`, `bld_dim_20`, `socio_demog_082`,
  `env_climate_05`). Karena EUI menyumbang sekitar 65% skor, target ini menjadi prioritas.
- Rata-rata target per zona hampir tidak membantu target makro (NRMSE gas 0,972). Rata-rata per subzona bahkan lebih
  buruk dari rata-rata global untuk gas (1,113), kecuali untuk adopsi (0,566). Sinyal kelompok baru berguna setelah
  target dinormalisasi ukuran, misalnya EUI atau kWh per meter.

## 6. Hubungan data latih dan data uji

| Bukti | Angka | Arti |
|---|---|---|
| Rasio baris latih | 0,700000 | dibuat dengan split acak 70:30 |
| Zona | 329 dari 330 zona muncul di data uji; 100% baris uji berada di zona yang dikenal | tidak ada zona baru |
| Subzona | 91,86% baris uji berada di subzona yang dikenal | baris subzona baru teramati 8,14% vs harapan 8,17% untuk split acak per baris |
| Rasio uji per zona | rata-rata 0,2993, std z-score binomial 0,989 | sebaran persis seperti split acak |
| *Adversarial validation* | AUC 0,4997 (tanpa `id`) | distribusi latih dan uji tidak dapat dibedakan |

**Implikasi.** Data uji adalah sampel acak 30% per baris. Validasi utama karenanya memakai KFold acak per baris,
dan informasi tingkat kelompok (zona, subzona, sel iklim) sah dipakai selama dihitung *out-of-fold*. Tidak ada
*drift* yang perlu dikoreksi dengan pembobotan ulang. Baris uji di subzona baru (8,1%) lebih sulit (NRMSE gas 0,281
vs 0,247 pada baris lain) dan didominasi subzona satu baris.

Figur: `reports/eda/figures/zone_test_ratio.png`, `reports/eda/figures/id_rolling_mean.png`.

## 7. *Noise* label dan batas akurasi

**Bukti.**
- Varians residual log mengikuti pola a + b/M, dengan M = jumlah meter. Untuk gas a = 0,0091 dan b = 0,377; untuk
  listrik a = 0,0113 dan b = 0,591.
- Komponen b/M berasal dari rata-rata konsumsi antarrumah tangga yang tidak tercatat di fitur. Komponen ini
  menyumbang sekitar 75% varians residual.
- Varians residual EUI sebanding dengan 1/luas, sesuai mekanisme yang sama.
- Pada adopsi, sisa galat mendekati *noise* binomial dari jumlah rumah tangga per OA (sekitar 120).
- Pencarian kebocoran tidak menemukan sinyal yang dapat dieksploitasi: urutan `id` acak, nilai gas kembar (sekitar
  2% baris) adalah kebetulan, dan jumlah target per subzona tidak cocok dengan fitur mana pun.

**Implikasi.** Bahkan bila galat model (a) bernilai 0, skor total tetap sekitar 0,213. Batas ini berasal dari
dataset, bukan dari model. Skor jauh di bawahnya pada *leaderboard* (papan peringkat) hanya mungkin diperoleh dengan
mencocokkan baris ke data sumber publik, yang dilarang oleh aturan kompetisi.

## 8. Rangkuman implikasi bagi pemodelan

1. Dua dari lima target (porsi bangunan bersih dan adopsi di subzona yang dikenal) selesai tanpa model; gas dan EUI
   berbagi satu model. Masalah ML tersisa: gas/EUI, listrik, dan adopsi di subzona baru.
2. Target makro berskala ukuran, sehingga diformulasikan per unit *exposure* (luas lantai, jumlah meter, jumlah EPC)
   dengan bobot yang menyamakan *loss* dengan MSE skala asli.
3. Split acak per baris membenarkan KFold acak dan statistik kelompok *out-of-fold*.
4. Data bersih; pembersihan cukup membuang 11 kolom konstan, meng-*encode* dua kolom teks, dan menyalin 77 target
   duplikat.
5. *Noise* label menetapkan lantai skor sekitar 0,21. Perbaikan yang tersisa kecil dan dicari melalui formulasi,
   ensemble, dan koreksi residual, bukan dari pembersihan data.
