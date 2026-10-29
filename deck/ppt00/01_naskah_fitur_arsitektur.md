# Naskah 01: Rekayasa Fitur dan Arsitektur Model

Sumber yang diperiksa:
- notebook `final/deliverables/DICEFEST2026_BISMILLAH.ipynb` (versi sudah dieksekusi, 56 sel);
- modul yang tertanam di notebook: `common.py`, `train_lib.py`, `features_knn.py`, `fb07b.py`, `adopt_v2.py`,
  `blend.py`, `postprocess.py`, `pipeline.py`.

Backup notebook sebelum perubahan: `notebook_backup.ipynb`. Hanya sel Markdown 28 dan 30 yang diubah (Bagian C).

Nomor sel dihitung dari 0. Di Jupyter yang terlihat adalah nomor eksekusi `In [n]`:

| Sel | Di Jupyter | Isi |
|---|---|---|
| 23 | In [21] | memuat modul; mencetak "42 anggota terdefinisi; 28 di blend gas/EUI, 13 di blend listrik" |
| 25 | In [22] | pemeriksaan identitas data |
| 29 | In [24] | membangun fitur (`Data()`), jumlah fitur per kelompok, korelasi |
| 31 | In [25] | diagram arsitektur dan tabel anggota |
| 35 | In [27] | model adopsi, penggabungan, koreksi, berkas submisi |
| 37 | In [28] | galat jujur per kolom dan tabel perkembangan skor |
| 45 | In [32] | lima belas besar fitur menurut rata-rata \|SHAP\| (ditampilkan 10) |
| 47 | In [33] | grafik dependensi SHAP |
| 54 | In [36] | Lampiran A: pemeriksaan pemisahan data |

## A. Hasil verifikasi terhadap deskripsi tugas

| Klaim di deskripsi | Status | Bukti |
|---|---|---|
| `area_per_meter` adalah fitur fisik andalan | Sesuai | `common.base_features`: `bld_dim_06 / bld_dim_12`; peringkat 1 SHAP untuk log EUI dan log listrik per meter (sel 45) |
| kNN-target *out-of-fold* dalam jendela `sub_n` | Sesuai | `train_lib.Data._knn_features` dan `features_knn.knn_stats`: tetangga dicari di antara baris dengan selisih `sub_n` ≤ W; pengaturan dasar (W, k) = (30, 20), (100, 20), (300, 50); fold dalam 5 bagian |
| Jalur gas/EUI 28 model, listrik 13 model | Sesuai | keluaran sel 23 |
| Jalur adopsi "2 LightGBM" | **Sebagian** | Jalur ketiga juga memuat rumus eksak porsi bangunan bersih. Adopsi disalin dari subzona yang dikenal (91,86% baris uji, sel 25). Dua LightGBM (`q_xent` dan `r_reg` di `adopt_v2.py`) hanya dipakai untuk subzona baru (8,14%), lalu dirata-rata (sel 35) |
| Diagram arsitektur di sel 31 | Sesuai | sel 31 = In [25], `report.architecture_diagram()` dan `report.member_table(cfg)` |

Catatan tambahan:
- **Jumlah model.** Ada 42 anggota terdefinisi, dan 41 di antaranya dipakai dalam gabungan. Anggota yang tidak
  dipakai adalah `mk1_gpm`, yaitu keluaran gas dari MLP gas + listrik.
- **Validasi model adopsi.** Validasinya memakai 5 fold per subzona. Hasilnya: RMSE semua subzona 0,8868, dan RMSE
  berbobot subzona baru 1,0303 poin persentase (sel 35).

## B. Naskah lisan (sekitar 650 kata, 5 menit)

### (a) Fitur ukuran rumah: `area_per_meter` (sekitar 1 menit 35 detik)

Kami mulai dari satu pertanyaan sederhana: mengapa satu kawasan memakai energi lebih banyak daripada kawasan lain?
Jawaban yang paling mudah adalah karena kawasannya lebih besar. Namun, jawaban itu tidak memberi tahu apa pun tentang
rumah di dalamnya. Karena itu kami membuat fitur area per meter, yaitu total luas lantai kawasan dibagi jumlah meteran
energinya. Satu meteran kira-kira mewakili satu rumah tangga, jadi angka ini menggambarkan luas rata-rata satu rumah.

Fitur sederhana ini ternyata paling berpengaruh. Pada model penjelas kami, area per meter menempati peringkat pertama,
baik untuk intensitas gas maupun untuk listrik per rumah. Arahnya berlawanan. Rumah kecil memakai gas per meter persegi
lebih banyak, karena dinding dan atapnya relatif luas dibanding ruang yang dipanaskan. Rumah besar justru memakai
listrik lebih banyak per rumah.

Apakah fitur ini membocorkan jawaban? Tidak. Rumusnya hanya membagi dua kolom input, luas lantai dan jumlah meter, yang
tersedia untuk setiap kawasan, baik di data latih maupun data uji. Tidak ada target yang dipakai, tidak ada rata-rata
yang dihitung lintas baris, dan rumus yang sama diterapkan baris demi baris. Kolom luas lantai memang juga dipakai
untuk mendefinisikan EUI, tetapi kolom itu diberikan panitia untuk setiap kawasan uji, jadi bukan informasi dari target.

### (b) Fitur kawasan termirip: kNN-target *out-of-fold* (sekitar 1 menit 40 detik)

Fitur kedua menjawab pertanyaan lain: berapa konsumsi kawasan tetangga yang paling mirip? Kami memakai kNN, singkatan
dari k nearest neighbors, yaitu metode mencari sejumlah k data yang paling mirip. Kemiripan diukur dari stok bangunan:
ukuran rumah, cakupan sertifikat energi, jumlah meter, dan komposisi periode bangunan. Pencarian dibatasi pada jendela
nomor subzona, misalnya 30, 100, atau 300 nomor di sekitarnya. Alasannya, nomor subzona ternyata berurutan secara
geografis: 99,07 persen baris yang berurutan nomornya berada di zona yang sama. Dari 20 sampai 50 tetangga terdekat
pada pengaturan dasarnya, kami mengambil rata-rata log intensitas gas, log listrik per meter, dan log gas per meter.

Karena fitur ini memakai target tetangga, ia wajib dihitung out-of-fold, artinya tanpa pernah melihat jawaban baris
yang sedang dinilai. Caranya berlapis. Data latih dibagi menjadi sepuluh bagian. Untuk baris validasi, tetangga hanya
diambil dari sembilan bagian lainnya. Untuk baris latih, bagian latih itu dibagi lagi menjadi lima, dan tetangga setiap
baris diambil dari empat bagian yang lain. Untuk data uji, tetangga diambil dari data latih yang dipakai model
tersebut. Jadi tidak ada satu baris pun yang melihat targetnya sendiri. Hasilnya terlihat di leaderboard publik: versi
yang memasukkan fitur ini ke dalam model menurunkan skor dari 0,24954 menjadi 0,24943.

### (c) Arsitektur tiga jalur (sekitar 1 menit 45 detik)

Lima target tidak kami paksakan ke satu model. Metrik lomba menilai setiap kolom secara terpisah, sehingga setiap
target boleh memakai cara terbaiknya sendiri. Kami membaginya menjadi tiga jalur.

Jalur pertama untuk gas dan EUI, berisi 28 model. Keduanya sebenarnya satu besaran, karena gas sama dengan EUI dikali
luas lantai, dengan selisih terbesar hanya 6,7 kali 10 pangkat minus 16. Jadi satu keluarga model melayani dua kolom.
Bedanya hanya pada bobot penggabungan: untuk kolom gas, kesalahan di kawasan besar dihitung lebih berat.

Jalur kedua untuk listrik, berisi 13 model. Listrik dipisah karena pendorongnya berbeda. Pada model penjelas, porsi
kontribusi sosiodemografi untuk listrik lebih besar daripada untuk gas.

Jalur ketiga hampir tidak membutuhkan model. Porsi bangunan bersih dihitung langsung dengan rumus, dan rumus itu tepat
di seluruh baris latih. Adopsi pemanas listrik seragam di dalam satu subzona, sehingga 91,86 persen kawasan uji cukup
disalin dari subzonanya. Untuk 8,14 persen sisanya, yaitu kawasan di subzona baru, kami memakai dua model LightGBM.
Model pertama menebak peluang bahwa rumah tangga di luar kategori pemanas yang tercatat bukan pengguna pemanas listrik.
Model kedua menebak porsi itu secara langsung. Hasil keduanya dirata-rata.

Model di dua jalur pertama sengaja beragam: XGBoost dan LightGBM yang berbasis pohon keputusan, jaringan saraf, dan
NCA yang menebak dari kawasan mirip. Totalnya 41 model, digabung dengan bobot tidak negatif yang berubah menurut luas,
jumlah meter, dan besar tebakan. Seluruh proses menghasilkan berkas submisi yang identik setiap kali dijalankan ulang.

## C. Usulan format sel Markdown

Kedua usulan di bawah sudah diterapkan ke notebook dan ke generatornya (`final/notebook_text.py`, dipakai oleh
`final/make_notebooks.py`).
- **Sel yang berubah:** hanya sel Markdown 28 dan 30. Jumlah sel tetap 56, jadi nomor sel lain tidak bergeser.
- **Pemeriksaan:** ke-37 sel kode beserta keluarannya identik dengan `notebook_backup.ipynb`.

### C.1 Bagian `common.base_features` (sel 28, judul bagian 2)

Bagian 2 kini dibuka dengan tabel fitur turunan, lalu dua kelompok lain:

```markdown
### 2.1 Fitur turunan tanpa target (`common.base_features`)
Setiap fitur di tabel berikut dihitung baris demi baris dari kolom input kawasan itu sendiri. Tidak ada target dan tidak
ada statistik lintas baris yang dipakai, dan rumus yang sama diterapkan pada data latih dan data uji, sehingga fitur ini
tidak dapat membocorkan target. Fitur terpenting di kelompok ini adalah `area_per_meter` (lihat bagian 7).

| Fitur | Rumus | Makna (dugaan, karena nama kolom disamarkan) |
|---|---|---|
| `area_per_meter` | `bld_dim_06` / `bld_dim_12` | luas lantai per meter, proksi ukuran rata-rata rumah |
| `area_per_epc` | `bld_dim_06` / `bld_dim_01` | luas lantai per sertifikat energi |
| `meters_per_epc` | `bld_dim_12` / `bld_dim_01` | meter per sertifikat energi |
| `clean_ratio` | `bld_dim_02` / `bld_dim_01` | porsi bangunan bersih (dasar rumus target keempat) |
| `gea_ratio` | `bld_dim_06` / `bld_dim_03` | luas lantai terhadap `bld_dim_03` (diduga luas selubung) |
| `a04_ratio`, `a05_ratio` | `bld_dim_04` / `bld_dim_03`, `bld_dim_05` / `bld_dim_03` | dimensi bangunan relatif terhadap `bld_dim_03` |
| `a07_ratio`, `a08_ratio` | `bld_dim_07` / `bld_dim_06`, `bld_dim_08` / `bld_dim_06` | dimensi bangunan relatif terhadap luas lantai |
| `s002_ratio` | `socio_demog_002` / `bld_dim_06` | kolom sensus 002 per luas lantai |
| `hvac_rest` | 100 − jumlah `hvac_dist_01` sampai `hvac_dist_10` | sisa porsi jenis pemanas (listrik dan lainnya) |

### 2.2 Fitur yang memakai target kawasan lain, selalu *out-of-fold*
(rata-rata target wilayah dan kawasan termirip, beserta penjelasan cara menghitungnya tanpa melihat target baris
itu sendiri)

### 2.3 Bentuk target dan jumlah rumah tangga
(target per rumah tangga dan jumlah rumah tangga N untuk adopsi)
```

Alasan format ini:
- **Rumus langsung terlihat.** Juri bisa mencocokkan setiap fitur dengan rumusnya tanpa membuka sel kode yang
  diciutkan.
- **Status kebocoran tiap kelompok tegas.** Kelompok 2.1 aman karena tidak memakai target, sedangkan kelompok 2.2
  memakai target kawasan lain dan karena itu dihitung *out-of-fold*.
- **Sepuluh baris pertama tabel cocok dengan keluaran sel 29,** yang menghitung 10 fitur "rasio fisik turunan".
  Sisanya, `hvac_rest`, masuk kelompok "bauran pemanas" bersama 10 kolom `hvac_dist`.

### C.2 Diagram di sel 31 (ditambahkan di akhir sel 30, tepat sebelum diagram)

```markdown
**Cara membaca diagram di bawah.**
1. Kotak abu-abu di kiri adalah data mentah (`train.csv` dan `test.csv`, 201 kolom fitur).
2. Jalur atas: fitur domain (biru), lalu 41 model, yaitu 28 untuk gas/EUI dan 13 untuk listrik (oranye), lalu tiga
   tahap berwarna kuning: gabungan dengan bobot yang menyesuaikan, koreksi galat wilayah, dan penyesuaian akhir.
3. Jalur bawah (hijau) tidak memakai model besar: porsi bangunan bersih dihitung dengan rumus dan adopsi disalin per
   subzona, ditambah model peluang untuk subzona baru. Hasilnya langsung masuk ke berkas submisi.
4. Teks miring di tengah mengingatkan bahwa gas = EUI × luas lantai, sehingga satu keluarga model melayani dua kolom.
5. Tabel di bawah diagram berisi satu baris per anggota gabungan: nama, jenis model, keluarga, besaran yang ditebak,
   bobot baris saat pelatihan, dan kelompok fitur yang dipakai.
```

Alasan format ini:
- **Diagram di sel 31 tidak berlabel nomor.** Daftar bernomor yang mengikuti warna kotak membantu pembaca non-teknis
  mengikuti alurnya.
- **Tabel anggota di bawah diagram punya enam kolom tanpa penjelasan.** Butir 5 menerangkan arti setiap kolomnya.
