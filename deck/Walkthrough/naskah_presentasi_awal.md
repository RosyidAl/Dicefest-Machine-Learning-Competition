# Naskah Presentasi Final: Skema Awal

Dokumen ini memuat skema awal dan naskah awal presentasi final (Zoom, 30 Oktober 2026), diambil dari catatan pembicara `deck/ppt04/BISMILLAH_Final_v5.pptx`. Isinya 13 *slide* utama dengan durasi total 9 menit 30 detik. Cadangan C1 sampai C12 hanya didaftar di bagian akhir.

Status naskah awal:
- Belum dilatih dengan pengatur waktu.
- Nama pembicara belum diisi, jadi dipakai penanda Orang 1 (jangkar), Orang 2 (teknis), dan Orang 3 (kebijakan).
- Tampilan *deck* belum diperiksa di PowerPoint asli.
- Bila naskah di *deck* berubah, catatan pembicara di PPTX yang menjadi acuan, bukan dokumen ini.

## Daftar isi

- [Skema waktu dan pembicara](#skema-waktu-dan-pembicara)
- [Naskah per *slide*](#naskah-per-slide)
  - [I. Ringkasan eksekutif](#i-ringkasan-eksekutif)
  - [II. Rekayasa fitur dan pra-pemrosesan](#ii-rekayasa-fitur-dan-pra-pemrosesan)
  - [III. Arsitektur model dan hasil](#iii-arsitektur-model-dan-hasil)
  - [IV. Rekomendasi kebijakan publik](#iv-rekomendasi-kebijakan-publik)
  - [V. Keterbatasan dan penutup](#v-keterbatasan-dan-penutup)
- [Slide cadangan untuk tanya jawab](#slide-cadangan-untuk-tanya-jawab)
- [Hal yang perlu diputuskan sebelum latihan](#hal-yang-perlu-diputuskan-sebelum-latihan)

## Skema waktu dan pembicara

Pesan utama (tesis): transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan. Tiga rekomendasi mengikuti tesis itu: *retrofit* di Q1 (R1), penilaian gardu di Q1 dan Q3 (R2), pilot pompa panas di Q2 (R3).

| Bagian | Durasi | Slide | Pembicara |
|---|---|---|---|
| I. Ringkasan eksekutif | 60 detik | 1 sampai 2 | Orang 1 (jangkar) |
| II. Rekayasa fitur dan pra-pemrosesan | 150 detik | 3 sampai 5 | Orang 2 (teknis) |
| III. Arsitektur model dan hasil | 90 detik | 6 sampai 7 | Orang 2 (teknis) |
| IV. Rekomendasi kebijakan publik | 210 detik | 8 sampai 11 | Orang 3 (kebijakan) |
| V. Keterbatasan dan penutup | 60 detik | 12 sampai 13 | Orang 1 (jangkar) |

| Slide | Waktu | Durasi | Pembicara | Judul | Kata naskah |
|---|---|---|---|---|---|
| 1 | 00:00–00:25 | 25 detik | Orang 1 (jangkar) | Trilema energi kota: elektrifikasi penuh hampir menggandakan konsumsi listrik zona | 44 |
| 2 | 00:25–01:00 | 35 detik | Orang 1 (jangkar) | Transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan | 61 |
| 3 | 01:00–01:55 | 55 detik | Orang 2 (teknis) | Luas lantai per meter: fitur teratas model penjelas konsumsi, dihitung tanpa menyentuh target | 95 |
| 4 | 01:55–02:45 | 50 detik | Orang 2 (teknis) | Fitur kawasan termirip memakai target tetangga, tetapi dihitung out-of-fold dua lapis | 86 |
| 5 | 02:45–03:30 | 45 detik | Orang 2 (teknis) | Uji label acak lulus: label baris yang dinilai tidak terbaca, versi bocor tertangkap | 79 |
| 6 | 03:30–04:30 | 60 detik | Orang 2 (teknis) | Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik | 106 |
| 7 | 04:30–05:00 | 30 detik | Orang 2 (teknis) | Skor publik turun di setiap langkah, dan skor privat dari data tersembunyi tidak memburuk | 52 |
| 8 | 05:00–05:55 | 55 detik | Orang 3 (kebijakan) | Peta 330 zona: 118 zona Q1 memuat 47% konsumsi gas, jadi program diurutkan per kuadran | 93 |
| 9 | 05:55–06:45 | 50 detik | Orang 3 (kebijakan) | Rekomendasi 1: retrofit selubung lebih dulu di zona Q1, mulai dari audit energi | 83 |
| 10 | 06:45–07:35 | 50 detik | Orang 3 (kebijakan) | Rekomendasi 2: kapasitas gardu di zona Q1 dan Q3 dinilai sebelum subsidi pompa panas dibuka | 89 |
| 11 | 07:35–08:30 | 55 detik | Orang 3 (kebijakan) | Rekomendasi 3: elektrifikasi kawasan bergas dimulai sebagai pilot pompa panas di zona Q2 | 95 |
| 12 | 08:30–09:05 | 35 detik | Orang 1 (jangkar) | Rekomendasi ini dasar prioritas, bukan pengganti audit teknis atau validasi lapangan | 60 |
| 13 | 09:05–09:30 | 25 detik | Orang 1 (jangkar) | Diurutkan per kuadran: retrofit dulu di Q1, gardu dinilai di Q1 dan Q3, pilot pompa panas di Q2 | 44 |

Total naskah yang diucapkan 987 kata untuk 570 detik, sekitar 103,9 kata per menit.

## Naskah per *slide*

### I. Ringkasan eksekutif

**Slide 1. Trilema energi kota: elektrifikasi penuh hampir menggandakan konsumsi listrik zona**

*00:00–00:25 (25 detik), Orang 1 (jangkar)*

> Bapak dan Ibu juri, kami tim BISMILLAH. Kota yang beralih dari pemanas gas ke pemanas listrik menghadapi trilema: emisi harus turun, jaringan tetap andal, dan biaya tetap terjangkau. Bila semua pemanas gas diganti pompa panas, median kenaikan konsumsi listrik tahunan zona mencapai 94,85 persen.

**Slide 2. Transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan**

*00:25–01:00 (35 detik), Orang 1 (jangkar)*

> Tesis kami: transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan. Pertama, demi biaya dan emisi, retrofit selubung didahulukan di 118 zona Q1 yang memuat 47 persen konsumsi gas. Kedua, demi keandalan, gardu di zona Q1 dan Q3 dinilai sebelum subsidi pompa panas dibuka. Ketiga, agar transisi terukur, kawasan bergas zona Q2 mulai dengan pilot. Rekan kami akan menunjukkan dasar analisisnya.

### II. Rekayasa fitur dan pra-pemrosesan

**Slide 3. Luas lantai per meter: fitur teratas model penjelas konsumsi, dihitung tanpa menyentuh target**

*01:00–01:55 (55 detik), Orang 2 (teknis)*

> Dasar pertama: fitur. Fitur utamanya sederhana: luas lantai kawasan dibagi jumlah meteran. Satu meteran kira-kira satu rumah tangga, jadi ini mendekati luas satu rumah. Fitur yang diam-diam membawa jawaban membuat skor tampak bagus, tetapi gagal di data baru. Risiko itu dikendalikan sejak desain: fitur ini hanya membagi dua kolom input baris yang sama, tanpa target, dengan rumus sama di data latih dan uji. Pada model penjelas, model terpisah yang dibaca dengan SHAP, rasio fisik turunan menyumbang 25 sampai 30 persen, dan fitur ini peringkat pertama, 2,8 dan 3,0 kali fitur kedua. Inilah dasar rekomendasi retrofit.

**Slide 4. Fitur kawasan termirip memakai target tetangga, tetapi dihitung out-of-fold dua lapis**

*01:55–02:45 (50 detik), Orang 2 (teknis)*

> Fitur kedua bersifat spasial. Untuk setiap kawasan, kami mencari kawasan dengan stok bangunan paling mirip di sekitarnya, lalu memakai rata-rata konsumsinya. Kedekatan diukur dari nomor subzona: 96,5 persen pasangan subzona bernomor berurutan berada di zona yang sama, jadi penomoran itu menjadi pendekatan lokasi. Karena memakai target tetangga, fitur ini berisiko bocor sehingga dihitung out-of-fold dua lapis. Lapis luar: baris yang dinilai hanya melihat tetangga dari sembilan fold lain. Lapis dalam: baris latih hanya melihat tetangga dari empat bagian lain, jadi tidak ada baris memakai targetnya sendiri.

**Slide 5. Uji label acak lulus: label baris yang dinilai tidak terbaca, versi bocor tertangkap**

*02:45–03:30 (45 detik), Orang 2 (teknis)*

> Desain itu kami uji, bukan kami asumsikan. Label 42.931 baris validasi kami acak, lalu fitur dihitung ulang: fitur validasi dan kawasan uji tetap identik. Pada baris latih, label bagian dalam diacak: fitur bagian itu identik, bagian lain berubah karena memakainya sebagai tetangga, sesuai rancangan. Versi yang sengaja dibuat bocor berubah, jadi uji ini mampu menangkap kebocoran. Cakupannya dua blok fitur pada satu fold. Fitur ini menurunkan galat EUI model tunggal, dan prediksi untuk 30 persen nilai peta zona memakainya.

- Bila ditanya (tidak diucapkan): angka kontribusi ini −0,0022 pada galat EUI model tunggal (notebook bagian 2), diukur sebagai tambahan di atas model yang sudah memuat rata-rata target wilayah dan agregat kawasan. Kata "terbesar" tidak dipakai karena dalam ablasi pengembangan, rata-rata target wilayah memberi perbaikan setara atau lebih besar.

### III. Arsitektur model dan hasil

**Slide 6. Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik**

*03:30–04:30 (60 detik), Orang 2 (teknis)*

> Arsitekturnya mengikuti tiga fakta data. Pertama, di data latih, gas selalu sama dengan EUI dikali luas lantai, jadi keduanya ditangani satu jalur dengan 28 model. Kedua, porsi bangunan bersih tepat sama dengan satu rumus di 100 persen baris latih, jadi tanpa model. Ketiga, adopsi seragam per subzona, jadi adopsi 91,86 persen kawasan uji disalin dan 8,14 persen sisanya ditebak 2 LightGBM. Listrik punya jalur sendiri dengan 13 model. Model tiap jalur digabung berbobot adaptif, dikoreksi galat tetangga, lalu disesuaikan dengan batas data. Galat di kanan dihitung sebelum koreksi. Totalnya 43 model: 41 di-blend dan 2 terpisah. Semua ada di notebook 56 sel dengan satu sakelar pelatihan.

**Slide 7. Skor publik turun di setiap langkah, dan skor privat dari data tersembunyi tidak memburuk**

*04:30–05:00 (30 detik), Orang 2 (teknis)*

> Apakah arsitektur ini bekerja di data baru? Skor publik turun di setiap langkah. Skor privat dari data tersembunyi 0,24844, peringkat 8, tidak memburuk. Validasi meniru kondisi uji karena data latih dan uji tidak dapat dibedakan: AUC 0,4997. Berkas submisinya identik setiap diulang di perangkat yang sama. Rekan kami akan melanjutkan ke kebijakan.

### IV. Rekomendasi kebijakan publik

**Slide 8. Peta 330 zona: 118 zona Q1 memuat 47% konsumsi gas, jadi program diurutkan per kuadran**

*05:00–05:55 (55 detik), Orang 3 (kebijakan)*

> Sekarang kebijakan. Nilai aktual kawasan latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Ini peta kuadran, bukan peta geografis: sumbu mendatar intensitas gas zona, sumbu tegak kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas. Garis median membagi empat kuadran. Kuadran satu berisi 118 zona di atas median pada kedua sumbu dan memuat 47 persen konsumsi gas. Penilaian gardu menyasar kuadran satu dan tiga, yang kenaikan listriknya di atas median. Urutan zona tahan derau prediksi acak, tetapi berubah bila COP pompa panas berbeda per iklim. Jadi, peta ini daftar kandidat.

**Slide 9. Rekomendasi 1: retrofit selubung lebih dulu di zona Q1, mulai dari audit energi**

*05:55–06:45 (50 detik), Orang 3 (kebijakan)*

> Rekomendasi pertama menyangkut retrofit. Temuan SHAP pada model penjelas: kawasan rumah terkecil punya sumbangan gas per meter persegi sekitar 65 persen lebih tinggi daripada rumah terbesar, dan satu periode bangunan sekitar 7 persen. Ini asosiasi, bukan sebab-akibat; kemungkinan karena dinding dan atap rumah kecil relatif luas. Temuan model mendukung rekomendasi ini: retrofit selubung lebih dulu di zona kuadran satu. Karena potensinya baru indikator, langkah pertamanya audit energi di REG_273, REG_270, dan REG_045, lalu pilot dengan pembanding. Pelaksananya pemerintah kota bersama penyedia jasa retrofit.

- Bila ditanya (tidak diucapkan): +65% di notebook bagian 7.1 adalah pembulatan dari 66,5% (deck/ppt01/bukti/cek_kebijakan.json); notebook menamai pitanya sekitar 30 dan 300 m² per meter (titik tengah interval), sedangkan grafik memakai rata-rata pita ±37 dan ±173 m².

**Slide 10. Rekomendasi 2: kapasitas gardu di zona Q1 dan Q3 dinilai sebelum subsidi pompa panas dibuka**

*06:45–07:35 (50 detik), Orang 3 (kebijakan)*

> Rekomendasi kedua menyangkut keandalan jaringan. Temuan SHAP: ukuran rumah adalah fitur teratas untuk listrik per rumah, dan sumbangannya di rumah besar sekitar 75 persen lebih tinggi. Bila COP berbeda per iklim, peringkat kenaikan persen berubah: REG_290 turun dari 1 ke 47. Peringkat tambahan absolut teratas stabil, jadi titik awal dipilih menurut nilai absolut: REG_263, REG_045, dan REG_270. Ini energi tahunan; beban puncak dan kapasitas gardu tidak ada di data. Temuan model mendukung rekomendasi ini sebagai daftar skrining: gardu dinilai per zona sebelum subsidi. Pelaksananya operator jaringan dan pemerintah kota.

**Slide 11. Rekomendasi 3: elektrifikasi kawasan bergas dimulai sebagai pilot pompa panas di zona Q2**

*07:35–08:30 (55 detik), Orang 3 (kebijakan)*

> Rekomendasi ketiga menyangkut adopsi. Temuan SHAP: sumbangan porsi gas jaringan pada adopsi menjadi negatif di atas 80 persen. Tingkat adopsinya 9,46 persen pada gas jaringan 60 sampai 80 persen, tetapi 2,73 persen di atas 80 persen, padahal kelompok itu 70 persen kawasan. Porsi jenis pemanas berjumlah tetap, jadi ini pola komposisi, bukan sebab-akibat. Temuan model mendukung rekomendasi ini: pilot pompa panas di kuadran dua seperti REG_033, yang intensitas gasnya tinggi tetapi persentase kenaikan listriknya di bawah median. Pilot didahului cek gardu dan memakai pembanding. Pelaksananya pemerintah kota dan penyedia pompa panas. Rekan kami akan menutup.

### V. Keterbatasan dan penutup

**Slide 12. Rekomendasi ini dasar prioritas, bukan pengganti audit teknis atau validasi lapangan**

*08:30–09:05 (35 detik), Orang 1 (jangkar)*

> Ada tiga batas yang kami akui. Pertama, data tanpa beban puncak dan kapasitas gardu, jadi penilaian gardu tetap tugas operator. Kedua, COP lokal, ambang median, dan makna kolom belum pasti, jadi daftar zona hanya kandidat. Ketiga, SHAP bukan sebab-akibat dan potensi retrofit baru indikator, jadi perlu audit dan pilot. Rekomendasi ini dasar prioritas, bukan pengganti audit teknis atau validasi lapangan.

**Slide 13. Diurutkan per kuadran: retrofit dulu di Q1, gardu dinilai di Q1 dan Q3, pilot pompa panas di Q2**

*09:05–09:30 (25 detik), Orang 1 (jangkar)*

> Trilema tidak selesai dengan program seragam. Retrofit di kuadran satu menjawab biaya dan emisi, penilaian gardu di kuadran satu dan tiga menjaga keandalan, pilot di kuadran dua menguji transisi. Langkah pertama: audit energi, data gardu dari operator, dan satu pilot dengan pembanding. Terima kasih.

## Slide cadangan untuk tanya jawab

Dipakai hanya bila juri bertanya. Penjawab tertera di catatan pembicara tiap *slide* cadangan.

| Slide | Judul | Penjawab |
|---|---|---|
| C1 (slide 14) | Bobot adaptif A2xM2xp2 tidak memburukkan kolom mana pun pada skor jujur | Orang 2 |
| C2 (slide 15) | Koreksi galat tetangga menggeser prediksi sebagian kecil mengikuti galat tetangga | Orang 2 |
| C3 (slide 16) | Dua blok fitur bertarget utama diuji dengan label acak pada satu fold | Orang 2 |
| C4 (slide 17) | Rincian 43 model: 28 jalur gas/EUI, 13 jalur listrik, dan 2 LightGBM adopsi | Orang 2 |
| C5 (slide 18) | SHAP dibaca dari model penjelas terpisah dan hanya menunjukkan sumbangan pada prediksi | Orang 3 |
| C6 (slide 19) | Celah data dan rencana validasi lapangan | Orang 3 |
| C7 (slide 20) | Validasi acak per baris meniru data uji, sedangkan validasi per zona akan membuang fitur yang berguna | Orang 2 |
| C8 (slide 21) | Penyesuaian akhir menangani dua batas data, dan hasilnya identik setiap diulang di perangkat yang sama | Orang 2 |
| C9 (slide 22) | Adopsi di subzona baru dimodelkan sebagai peluang dari jumlah rumah tangga | Orang 2 |
| C10 (slide 23) | Urutan zona tahan derau acak; zona contoh tetap pada kedua skenario COP | Orang 3 |
| C11 (slide 24) | Target efisiensi dan target emisi perlu dipisahkan | Orang 3 |
| C12 (slide 25) | Tidak ada pencilan yang dibuang | Orang 2 |

## Hal yang perlu diputuskan sebelum latihan

1. Nama pembicara untuk Orang 1, 2, dan 3.
2. Alasan memakai log target (di *deck* masih `[PERLU DIISI]`).
3. Jawaban untuk pertanyaan "mengapa REG_273 bukan titik awal R2" (alasan belum tertulis).
4. Pemeriksaan tampilan *deck* di PowerPoint asli.
5. Latihan satu putaran penuh dengan pengatur waktu 9:30.
