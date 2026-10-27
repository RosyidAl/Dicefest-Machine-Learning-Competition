# Jawaban Pertanyaan Persiapan Final (berdasarkan notebook)

Sumber: `final/deliverables/DICEFEST2026_BISMILLAH.ipynb`, versi yang sudah dieksekusi (56 sel). Jawaban hanya
memakai kode, keluaran sel, dan teks yang memang ada di notebook tersebut. Kode lengkapnya ada di sel-sel modul yang
diciutkan (`%%writefile dicefest_nb/src/...`) dan dirujuk dengan nama modulnya. Nomor sel di bawah dihitung dari 0.

## 1. Tema dan Target Prediksi

**Masalah utama.** Judul notebook adalah "DICE FEST 2026: Trilema Transisi Energi Perkotaan". Tugasnya memprediksi
lima besaran energi untuk setiap kawasan permukiman. Model yang sama kemudian dipakai untuk menyusun peta prioritas
kebijakan bagi 330 zona (bagian 7).

**Data** (keluaran sel 25): data latih 429.303 baris × 209 kolom, data uji 183.987 baris × 204 kolom.

**Lima target (multi-output):**

| Kolom | Arti | Σ (pembagi metrik, keluaran sel 27) |
|---|---|---|
| `target_macro_gas_kwh` | konsumsi gas total kawasan (kWh) | 151.750,04 |
| `target_macro_elec_kwh` | konsumsi listrik total kawasan (kWh) | 41.725,37 |
| `target_eui_gas_kwh_per_sqm` | intensitas gas per meter persegi lantai (EUI) | 40,65 |
| `target_clean_building_ratio` | porsi bangunan bersih (%) | 12,49 |
| `target_electric_heating_adoption` | adopsi pemanas listrik (%) | 7,18 |

**Metrik.** Galat RMSE tiap target dibagi σ di atas, lalu dirata-rata.

**Temuan yang menentukan cara menangani tiap target** (keluaran sel 25):
- Gas = EUI × `bld_dim_06` (luas lantai), dengan galat relatif maksimum 6,7e-16.
- Porsi bangunan bersih = round(100 × `bld_dim_02` / `bld_dim_01`, 2), tepat di 100% baris latih.
- Adopsi konstan di 100% subzona, dan 91,86% baris uji berada di subzona yang ada di data latih.
- Porsi data uji tepat 0,300000, yaitu pembagian acak 70/30 per baris.

## 2. Rekayasa Fitur Terkuat (25 poin)

Fitur-fitur ini dibangun di `common.base_features`, `te.py`, `features_knn.py`, `fb07b.py`, dan
`train_lib.Data.agg_features`. Keluaran sel 29 mencatat jumlah fitur per kelompok: sosio-demografi 118, dimensi
bangunan 43, agregat kawasan 16, era bangunan 14, bauran pemanas 11, rasio fisik turunan 10, iklim 5, lokasi 2.

**A. Fitur fisik andalan: ukuran rumah (`area_per_meter`).**
- Dihitung sebagai `bld_dim_06 / bld_dim_12`, yaitu luas lantai dibagi jumlah meter, kira-kira luas rata-rata satu
  rumah.
- Ini fitur dengan kontribusi SHAP terbesar untuk EUI maupun listrik per meter (sel 45).
- Korelasinya dengan log EUI -0,347 (sel 29).
- Rasio fisik lain di modul yang sama:
  - `area_per_epc` (luas per sertifikat energi);
  - `meters_per_epc` (meter per sertifikat);
  - `gea_ratio` (`bld_dim_06 / bld_dim_03`);
  - `clean_ratio` (`bld_dim_02 / bld_dim_01`);
  - `s002_ratio` (`socio_demog_002 / bld_dim_06`);
  - `a04/a05/a07/a08_ratio`;
  - `hvac_rest`, yaitu 100 dikurangi jumlah porsi jenis pemanas yang tercatat. Korelasinya dengan adopsi 0,988
    (sel 29).

**B. Fitur spasial andalan: kawasan termirip (kNN-target).**
- Fitur ini dibuat di `train_lib.Data.knn_features` dan `features_knn.knn_stats`, dengan tambahan di `fb07b.py`.
- Untuk setiap kawasan dicari k kawasan latih yang stok bangunannya paling mirip, di dalam jendela nomor subzona
  (`sub_n`) yang berdekatan.
- Ruang kemiripannya terdiri dari:
  - log(luas/meter), log(sertifikat/meter), dan log(meter);
  - porsi bangunan era 09 sampai 15;
  - `bld_dim_02`/sertifikat, log1p(`bld_dim_17`), dan log(`bld_dim_03`/luas).
- Nilai yang diambil adalah rata-rata log EUI, log listrik per meter, dan log gas per meter dari tetangga tersebut.
- Pengaturan dasarnya (W, k, bobot jarak) adalah (30, 20, 0), (100, 20, 4), dan (300, 50, 0). Semua dihitung
  *out-of-fold* dengan 5 fold di dalam data latih.
- `fb07b.py` menambah beberapa sudut pandang lain: ruang era bangunan, ruang bangunan + PCA sensus, jendela sangat
  lokal (W = 10), dan jendela lebar.
- Dampaknya terlihat di tabel perkembangan skor (sel 37): versi v16 "fitur kawasan paling mirip di dalam model"
  menurunkan skor LB publik dari 0,24954 ke 0,24943.

**C. Rata-rata target wilayah secara out-of-fold** (`te.py`, `train_lib.Data.te_features`).
- Rata-rata target dihitung di subzona, zona, petak iklim, dan jendela `sub_n`.
- Petak iklim adalah gabungan `env_climate_01/03/05`; komentar kode menyebut sekitar 15 baris per petak.
- Penggunaan jendela `sub_n` didukung Lampiran A (sel 54): 99,07% baris yang diurutkan menurut nomor subzona berada
  di zona yang sama dengan baris sebelumnya.

**D. Agregat kawasan.** Ada 16 fitur jumlah kawasan, luas, meter, dan sertifikat per subzona dan zona. Agregat ini
dihitung dari data latih + uji tanpa target.

**Kesimpulan.** Dua andalan utama tim adalah `area_per_meter` (fisik) dan fitur kawasan termirip (spasial).

## 3. Arsitektur Model (15 poin)

Multi-output tidak ditangani oleh satu model tunggal, melainkan oleh tiga jalur. Diagramnya ada di sel 31
(`report.architecture_diagram`).

1. **Jalur gas/EUI.** Satu keluarga model melayani dua kolom, karena gas = EUI × luas lantai. Ada 28 model dalam
   gabungan gas/EUI (keluaran sel 23).
2. **Jalur listrik.** Terdiri dari 13 model dalam gabungan listrik.
3. **Jalur identitas dan adopsi.**
   - Porsi bangunan bersih dihitung dengan rumus.
   - Adopsi disalin dari subzona yang sudah dikenal.
   - Untuk subzona baru dipakai dua LightGBM: `q_xent` (model peluang) dan `r_reg` (regresi), lalu keduanya
     dirata-rata. CV-nya: RMSE semua 0,8868, dan RMSE berbobot subzona baru 1,0303 (sel 35).

**Algoritma dasar** (`member_table`, sel 31; konfigurasi `configs/members.yaml`):
- XGBoost (GPU) dan LightGBM;
- MLP, termasuk varian PLR/TabM-mini;
- satu MLP multi-output yang memprediksi gas dan listrik sekaligus (`mk1`/`mk2`);
- NCA (`nca.py`), yaitu jaringan yang memprediksi dari rata-rata kawasan mirip.

Secara total ada 42 anggota terdefinisi, dan 41 dipakai dalam gabungan (sel 23 dan 33). Tiap model memprediksi
target per rumah tangga (misalnya gas per meter atau per luas), lalu hasilnya dikalikan kembali.

**Penggabungan dan tahap akhir** (`pipeline.assemble`, sel 35):
1. NNLS (bobot tak negatif) dengan koefisien yang berubah menurut luas, jumlah meter, dan besar prediksi (spesifikasi
   `A2xM2xp2`).
2. Koreksi galat wilayah, dengan besar koreksi (rms/σ) gas 0,02047, EUI 0,04344, listrik 0,01195.
3. Penyesuaian akhir S1 (kawasan bermeter sedikit), S3 (batas gas), dan S4 (batas nilai data latih).
4. Penyalinan 77 kawasan duplikat.

**Hasil.**
- Galat jujur gabungan sebelum koreksi (sel 37): gas 0,24494, EUI 0,65023, listrik 0,31587.
- Berkas submisi yang dihasilkan memiliki sha256 `aec745ab...d66e`.
- LB publik akhir 0,24913.

## 4. Variabel Paling Berpengaruh (35 poin)

SHAP dihitung pada **model penjelas**: satu LightGBM per target, yang hanya memakai karakteristik kawasan terukur
(`explain.py`, sel 41). SHAP ini tidak berasal dari gabungan 41 model. Nilai SHAP dihitung dengan TreeSHAP bawaan
LightGBM (`pred_contrib`) pada 20.000 baris sampel.

Kecocokan model penjelas pada data yang tidak dilihat (R², sel 41): log EUI 0,563, log gas per meter 0,747, log
listrik per meter 0,491, adopsi 0,991.

Lima variabel teratas menurut rata-rata |SHAP| (keluaran sel 45):

| Peringkat | log EUI | log listrik per meter | Adopsi pemanas listrik (%) |
|---|---|---|---|
| 1 | `area_per_meter` 0,0895 | `area_per_meter` 0,0895 | `hvac_rest` 3,5653 |
| 2 | `env_climate_05` 0,0321 | `meters_per_epc` 0,0294 | `hvac_dist_02` 0,9053 |
| 3 | `socio_demog_082` 0,0276 | `socio_demog_005` 0,0222 | `hvac_dist_09` 0,2691 |
| 4 | `area_per_epc` 0,0260 | `socio_demog_038` 0,0196 | `hvac_dist_01` 0,0968 |
| 5 | `s002_ratio` 0,0250 | `socio_demog_061` 0,0163 | `hvac_dist_08` 0,0953 |

Makna kolom ditulis di notebook sebagai dugaan karena nama kolom disamarkan:
- `area_per_meter`: luas lantai per meter atau ukuran hunian;
- `hvac_rest`: sisa porsi pemanas (listrik dan lain-lain);
- `hvac_dist_02`: gas jaringan (sentral);
- `hvac_dist_09`: dua jenis pemanas atau lebih (non-terbarukan);
- `hvac_dist_01`: tanpa pemanas sentral;
- `hvac_dist_08`: jaringan panas distrik;
- `meters_per_epc`: meter per sertifikat;
- `area_per_epc`: luas per sertifikat.

**Ringkasan variabel paling dominan:** `area_per_meter`, `hvac_rest`, `hvac_dist_02`, `env_climate_05`, dan
`meters_per_epc`.

Catatan penting untuk dibawakan:
- **Nilai SHAP tidak bisa dibandingkan antartarget**, karena satuannya berbeda (skala log untuk EUI dan listrik, poin
  persentase untuk adopsi).
- **Teks notebook menyebut kontribusi komposisi pemanas pada adopsi sebesar 81%** (bagian 7, di atas sel 43; angkanya
  hanya ada di teks, sedangkan grafik kelompok fitur berbentuk gambar). Angka ini sebagian besar adalah identitas
  akuntansi: makin kecil porsi pemanas lain, makin besar sisa untuk pemanas listrik. Jadi angka itu dibaca sebagai
  komposisi, bukan faktor pendorong.
- **Model gas per meter juga dilatih,** tetapi daftar fitur teratasnya tidak dicetak. Sel 45 hanya menampilkan EUI,
  listrik per meter, dan adopsi.

## 5. Kelemahan Pemahaman Kode

Bagian yang paling mungkin sulit dipahami tim ada di tahap setelah model dilatih: modul `blend.py`, `postprocess.py`,
dan `pipeline.assemble` (dijalankan di sel 35). Kodenya padat, banyak memakai indeks dan matriks, dan sebagian
angkanya dibekukan di `configs/resid_knn.json`. Fitur kNN `out-of-fold` (`features_knn.py`, `fb07b.py`) berada di
urutan kedua. Sebaliknya, notebook ini tidak memuat *hyperparameter tuning*: semua hiperparameter sudah ditetapkan di
`configs/members.yaml`. Bagian SHAP relatif sederhana karena hanya memanggil `model.predict(..., pred_contrib=True)`.

**a. Penggabungan dengan bobot yang berubah (spesifikasi `A2xM2xp2` di `blend.py`).**
- **Cara sederhana memahaminya:** setiap model memberi tebakan, lalu tebakan itu dicampur dengan bobot yang tidak
  boleh negatif (NNLS).
- **Yang membuatnya rumit:** bobotnya tidak tetap. Kawasan diurutkan menurut luas (A), jumlah meter (M), dan besar
  tebakan (p). Untuk tiap variabel dibuat dua "tombol" yang naik turun secara halus mengikuti peringkat tersebut, lalu
  tombol-tombol itu dikalikan.
- **Hasilnya:** campuran model untuk kawasan kecil bisa berbeda dari campuran untuk kawasan besar. Alasannya, model
  yang paling akurat di kawasan kecil belum tentu paling akurat di kawasan besar.

**b. Koreksi galat wilayah (`postprocess.py`).**
- Setelah dicampur, tebakan setiap kawasan digeser sedikit mengikuti rata-rata kesalahan tetangganya, yaitu kawasan
  di zona dan petak iklim yang sama, nomor subzona berdekatan, dan k kawasan termirip.
- **Contoh:** kalau tetangga-tetangga yang mirip ternyata selalu ditebak terlalu rendah, tebakan kawasan ini ikut
  dinaikkan sedikit.
- Rata-rata tetangga "diciutkan" bila tetangganya sedikit (`n / (n + K)` dengan K = 30), supaya satu tetangga yang
  aneh tidak menggeser terlalu jauh.
- Besar geseran (λ) tidak dihitung di notebook, melainkan diambil dari `resid_knn.json`. Nilainya ditentukan
  sebelumnya pada 30% data latih yang disisihkan (dijelaskan di teks bagian 3).

**c. Penyesuaian akhir S1, S3, S4 (`pipeline.py`, blok post-processing).**
- S1 menaikkan tebakan kawasan bermeter 2 sampai 4 dengan faktor tertentu, karena kawasan seperti ini selalu ditebak
  terlalu rendah.
- S3 menarik tebakan gas yang mendekati batas satu juta kWh ke bawah dengan bentuk distribusi normal terpotong,
  karena data latih tidak pernah melewati batas itu.
- S4 memotong semua tebakan ke rentang nilai yang ada di data latih.

**d. Fitur kNN out-of-fold (`features_knn.knn_stats`, `fb07b.py`).**
- **Cara sederhananya:** untuk setiap kawasan, cari 20 sampai 50 kawasan dengan nomor subzona yang dekat dan stok
  bangunan paling mirip, lalu ambil rata-rata konsumsinya.
- **Yang membuatnya rumit:** saat melatih, kawasan itu sendiri dan bagian datanya tidak boleh ikut dihitung.
  Karena itu data latih dibagi lagi menjadi 5 bagian di dalam setiap fold. Kodenya juga memakai teknik pencarian
  cepat (data diurutkan menurut `sub_n`, lalu diproses per blok) sehingga sulit dibaca.
- **Intinya tetap sederhana:** "kawasan yang mirip dan berdekatan cenderung memakai energi yang mirip", dihitung
  tanpa mencontek jawaban kawasan itu sendiri.

**Saran untuk tanya jawab.** Tim cukup menguasai alur besarnya dan alasannya; baris kodenya tidak perlu dihafal.
- Mengapa dicampur dengan bobot yang berubah: karena akurasi model berbeda menurut ukuran kawasan.
- Mengapa dikoreksi dengan galat tetangga: karena galat masih berkorelasi antarkawasan yang berdekatan.
- Mengapa ada penyesuaian akhir: karena ada dua batas di data, yaitu kawasan bermeter sedikit dan batas gas.
- Dari mana angka koreksinya: dari data latih 30% yang disisihkan, bukan dari leaderboard.
