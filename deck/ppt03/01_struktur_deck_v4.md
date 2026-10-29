# Struktur *Deck* Final v4: *Top-Down* dengan Prinsip Piramida Minto

Dokumen ini adalah sumber isi *deck* v4 presentasi final DICE FEST 2026 Tim BISMILLAH (Zoom, 30 Oktober 2026). Versi
ini merevisi *deck* v3 (`../ppt02`) atas permintaan pengguna tanggal 28 Oktober 2026. Beberapa elemen diambil dari
*deck* v2 (`../ppt01`), dan rinciannya per kriteria penilaian tercatat di bagian G4. Alurnya tetap *top-down*
(kesimpulan disampaikan lebih dahulu) menurut Prinsip Piramida Minto. Tesis dan tiga rekomendasi disampaikan pada
menit pertama, disusul dasar analisis (fitur dan arsitektur), bukti kebijakan, lalu keterbatasan. Durasi bicara tepat
9 menit 30 detik. Sisa 30 detik menjadi cadangan waktu (*buffer*) untuk serah terima antarpembicara, jeda, dan
gangguan teknis.

**Sumber angka.** Seluruh angka berasal dari keluaran *notebook* final `final/deliverables/DICEFEST2026_BISMILLAH.ipynb`
(`deck/fakta_notebook.txt`; nomor sel dihitung dari 0), `deck/fakta_turunan.md`, `deck/ppt01/fakta_tambahan.md`,
*writeup* B (`reports/WRITEUP_DRAFT_vB.md`), dan `deck/ppt02/fakta_ppt02.md`. Skrip `cek_struktur.py` memeriksa setiap
angka di *slide* utama dan *slide* cadangan terhadap kelima berkas tersebut. Baris kaki setiap *slide* utama memuat
nomor sel *notebook* asal angkanya (medan "Sel notebook").

**Format *slide*.** Setiap *slide* memuat empat komponen yang diminta, yaitu Judul Slide, Visual Utama, Isi Teks, dan
Narasi Pembicara, ditambah metadata durasi, bagian rubrik, dan sumber. Templat arahan memisahkan nomor *slide* dan
rentang waktu dengan tanda pisah panjang. Karena aturan gaya proyek melarang karakter tersebut, pemisahnya diganti
dengan titik tengah.

**Hubungan dengan *deck* sebelumnya.** Folder `deck/ppt01` (v2) dan `deck/ppt02` (v3) tetap utuh. Dokumen ini
dibangun menjadi `BISMILLAH_Final_v4.pptx` oleh `build_deck_v4.js`, yang membaca judul, isi layar, narasi, nomor sel,
dan bukti cadangan langsung dari dokumen ini melalui `isi_deck.py`. Gambar dan data grafiknya dibuat ulang oleh
`make_visuals_v4.py`. Cara membangun dan memeriksanya tercantum di `README.md`.

**Audit independen v3.** Versi pertama dokumen v3 diperiksa oleh pemeriksa berkonteks bersih yang menemukan 25 masalah
(2 berat, 13 sedang, 8 ringan, 2 dugaan). Setiap temuan diverifikasi ulang terhadap sumber sebelum diperbaiki;
perubahannya dirangkum di bagian G2.

## A. Identifikasi sebelum penyusunan *slide*

### A1. Tesis utama

> **Transisi energi 330 zona perlu diurutkan per kuadran, bukan diseragamkan: kurangi kebutuhan panas di zona boros
> gas, nilai gardu sebelum subsidi, dan buktikan elektrifikasi melalui pilot.**

Tesis ini merupakan bentuk ringkas rekomendasi pertama *notebook* sel 51 ("Program perlu diurutkan per kuadran, tidak
diseragamkan"). Inti yang perlu diingat juri cukup tiga kata: diurutkan, bukan diseragamkan. Kata "diurutkan" berarti
setiap kuadran mendapat program dan urutan sendiri, bukan satu urutan waktu untuk semua zona.

### A2. Tiga rekomendasi yang disampaikan pada menit pertama

| No. | Rekomendasi | Kuadran sasaran | Tuntutan trilema yang dijawab | Pelaksana | Asal di sumber |
|---|---|---|---|---|---|
| R1 | *Retrofit* (perbaikan selubung bangunan) lebih dulu di 118 zona Q1, dimulai dari audit energi di REG_273, REG_270, dan REG_045 | Q1 | biaya dan emisi | pemerintah kota dan penyedia jasa *retrofit* | sel 51 butir 1; *writeup* 7.4 butir 1 |
| R2 | Kapasitas gardu dinilai sebelum subsidi pompa panas dibuka, terutama di zona Q1 dan Q3 yang kenaikan konsumsi listriknya di atas median, dimulai dari REG_263, REG_270, dan REG_045 | Q1 dan Q3 | keandalan jaringan | operator jaringan distribusi dan pemerintah kota | sel 51 butir 3; `deck/ppt02/bukti/cek_zona_contoh.json` |
| R3 | Elektrifikasi kawasan bergas dimulai sebagai pilot pompa panas di zona Q2 yang tetap Q2 pada kedua skenario COP (contoh REG_033), dengan bantuan pemasangan dan pembongkaran ketel lama | Q2 | transisi yang terukur | pemerintah kota dan penyedia pompa panas | sel 51 butir 2 dan 4; `deck/ppt02/bukti/cek_zona_contoh.json` |

*Notebook* memuat lima rekomendasi. *Deck* ini memakai tiga rekomendasi yang merangkum butir 1 sampai 4. Butir 5
(pemisahan target efisiensi dan target emisi) disiapkan sebagai *slide* cadangan C11. Zona awal R2 dan contoh zona
R3 berbeda dari *notebook* dan *deck* v2 karena hasil uji ketahanan; alasannya tercatat di bagian E butir 14. Kolom
kuadran sasaran mengikuti sel 51 butir 3 ("operator jaringan perlu dilibatkan sejak awal pada zona Q1 dan Q3") dan
dipakai sama di *slide* 2, 8, 10, dan 13 (bagian E butir 21).

### A3. Bukti terkuat untuk setiap rekomendasi

| Rekomendasi | Bukti terkuat | Jenis bukti | Sumber |
|---|---|---|---|
| R1 | Q1 berisi 118 zona dengan 47% konsumsi gas dan indikator potensi *retrofit* 3,65 dari 7,39 TWh. Pada SHAP model penjelas, kawasan rumah terkecil memiliki sumbangan gas per m² sekitar +65% dibandingkan dengan rumah terbesar (antarkuartil +34%). Ketiga zona kandidat tetap Q1 pada kedua skenario COP | fakta peta, hasil model penjelas, uji ketahanan | sel 47 sampai 50; `cek_kebijakan.json`; `cek_zona_contoh.json` |
| R2 | Pada SHAP model penjelas, ukuran rumah adalah fitur teratas listrik per meter, dan kawasan rumah terbesar memiliki sumbangan listrik per rumah sekitar +75% dibandingkan dengan rumah terkecil: konsumsi listrik dasar sudah berbeda jauh antarkawasan. Median kenaikan konsumsi listrik tahunan zona +94,85%; lima zona tertinggi +113% sampai +121%. REG_270 dan REG_045 tetap sepuluh besar dalam kenaikan relatif maupun tambahan absolut, dan REG_263 tetap terbesar dalam tambahan absolut (0,76 TWh), pada kedua skenario COP. Data tidak memuat beban puncak dan kapasitas gardu | hasil model penjelas, hasil skenario dengan asumsi | sel 45, 47, 48, 50; `cek_kebijakan.json`; `cek_zona_contoh.json`; `img/data_grafik.json` |
| R3 | Tingkat adopsi pemanas listrik 40,24% di kawasan dengan porsi gas jaringan ≤40%, tetapi hanya 2,73% di kawasan >80% yang mencakup 70% kawasan; pola yang sama terlihat pada SHAP model penjelas | fakta data latih dan hasil model penjelas | sel 48; `cek_kebijakan.json` |
| Keandalan analisis | Uji label acak pada dua blok fitur bertarget: baris yang dinilai tidak terbaca dan kontrol bocor tertangkap; skor privat 0,24844 (peringkat 8); AUC adversarial 0,4997; berkas submisi identik saat diulang | hasil uji | `bukti/uji_kebocoran_*.json`; sel 35, 55 |

### A4. Angka dan definisi yang wajib dipertahankan

| Butir | Nilai dan definisi yang dipakai di seluruh *deck* | Sumber |
|---|---|---|
| Jumlah model | 43 model: 41 model masuk penggabungan berbobot adaptif (28 jalur gas/EUI dan 13 jalur listrik, dua kelompok terpisah), ditambah 2 LightGBM adopsi yang berdiri sendiri | sel 23, 33, 35 |
| 8,14% | porsi kawasan uji (baris data uji) yang berada di subzona tanpa baris latih; bukan porsi jumlah subzona | sel 35, 54 |
| 330 zona | unit lokasi tingkat atas (kode REG_xxx). Peta yang dipakai adalah peta kuadran, bukan peta geografis | sel 48, 49 |
| Q1, Q2, dan Q3 | Q1: intensitas gas zona (total gas dibagi total luas lantai) dan kenaikan konsumsi listrik tahunan sama-sama di atas atau sama dengan median 330 zona. Q2: intensitas di atas atau sama dengan median, kenaikan di bawah median. Q3: intensitas di bawah median, kenaikan di atas atau sama dengan median. Kenaikan di atas median berarti Q1 atau Q3 | `explain.zone_indices` (sel 19); sel 48 |
| Kenaikan konsumsi listrik | gas × 0,8 / 3 / listrik saat ini; energi tahunan, bukan beban puncak | sel 48, 51 |
| Potensi *retrofit* | indikator: gas yang berkurang bila sumbangan SHAP stok bangunan di atas rata-rata diturunkan ke rata-rata (7,39 TWh). Bukan janji penghematan; arah biasnya dua sisi (bagian E butir 4) | sel 48; `cek_kebijakan.json` |
| `area_per_meter` | `bld_dim_06` / `bld_dim_12`, yaitu luas lantai per meter atau kira-kira luas rata-rata satu rumah | sel 28 |
| Kedekatan nomor subzona | 96,5% pasangan subzona bernomor berurutan berada di zona yang sama, sehingga nomor yang berdekatan dipakai sebagai pendekatan kedekatan lokasi | `cek_zona_contoh.json`; sel 54 |
| kNN *out-of-fold* dua lapis | lapis luar 10 *fold* (bagian data); lapis dalam 5 bagian dari bagian latih | sel 28 |
| Galat jujur | gas 0,2449; EUI 0,6502; listrik 0,3159 (RMSE/σ, sebelum koreksi galat tetangga); makin kecil makin baik, dan menebak rata-rata bernilai sekitar 1 | sel 35, 39; *writeup* bab 1 |
| Skor | *leaderboard* (papan peringkat) privat 0,24844, peringkat 8; publik 0,24913 | Kaggle; sel 37 |

### A5. Keterbatasan data dan analisis yang diakui (sudah diverifikasi di sumber)

| Keterbatasan | Status di sumber | Bukti |
|---|---|---|
| Beban puncak aktual | tidak tersedia | sel 48: "Data kompetisi tidak memuat profil beban maupun kapasitas jaringan" |
| Kapasitas gardu terukur | tidak tersedia | idem |
| COP lokal | tidak tersedia; urutan zona peka bila COP berbeda per iklim (Spearman 0,81) | `cek_kebijakan.json` |
| Ambang kuadran | median dipilih demi kesederhanaan, bukan batas fisis | sel 51, paragraf Keterbatasan |
| Makna kolom | nama kolom disamarkan; makna seperti gas jaringan, jumlah meter, dan periode bangunan adalah dugaan, kecuali luas lantai yang terbukti dari identitas | sel 28, 40 |
| Nilai zona | 30% berasal dari prediksi kawasan uji yang cenderung menyusut ke arah rata-rata | *writeup* 7.3 |
| Hubungan sebab-akibat | tidak dapat disimpulkan dari SHAP | sel 51, paragraf Keterbatasan |
| Potensi *retrofit* yang nyata | 7,39 TWh hanya indikator; bila hanya periode bangunan dihitung tersisa sekitar separuh, sedangkan acuan yang lebih ketat dapat membuatnya lebih besar | `cek_kebijakan.json`; *writeup* 7.3 |
| Cakupan uji kebocoran | uji label acak mencakup dua blok fitur bertarget pada satu *fold* | `bukti/uji_kebocoran_*.py` |
| Koordinat geografis | tidak tersedia; data hanya memuat kode zona dan subzona | *writeup* bab 1 |
| Biaya *retrofit*, tarif, dan intensitas karbon listrik | tidak tersedia, sehingga manfaat ekonomi dan penurunan emisi bersih tidak dihitung | kelompok kolom data (*writeup* bab 1) |

### A6. Detail teknis yang hanya muncul di *slide* cadangan

Mekanisme bobot adaptif `A2xM2xp2` (C1), koreksi galat tetangga (C2), tabel lengkap uji label acak dan cakupannya
(C3), rincian keluarga model dan perbedaan 42 dengan 41 (C4), cara membaca SHAP dan batasnya (C5), celah data dan
rencana validasi lapangan (C6), perbandingan skema validasi (C7), penyesuaian akhir dan reprodusibilitas (C8), model
adopsi berbasis jumlah rumah tangga (C9), uji ketahanan dan zona contoh (C10), serta pemisahan target efisiensi dan
target emisi (C11).

## B. Piramida argumen dan peta waktu

### B1. Piramida argumen

```
TESIS     Transisi energi 330 zona perlu diurutkan per kuadran, bukan diseragamkan
  |
  +-- R1  Retrofit dulu di 118 zona Q1          biaya, emisi   bukti: peta (S8), SHAP gas per m² (S9)
  +-- R2  Gardu dinilai di Q1 dan Q3            keandalan      bukti: SHAP listrik per rumah, skenario (S8, S10)
  +-- R3  Pilot pompa panas di zona Q2          transisi       bukti: SHAP adopsi per porsi gas jaringan (S11)
  |
DASAR     Fitur bertarget dihitung out-of-fold dan diuji (S3 sampai S5); arsitektur teruji (S6, S7)
BATAS     Beban puncak, gardu, COP lokal, ambang dan makna kolom, sebab-akibat (S12); penutup (S13)
```

Uji membaca judul: bila juri hanya membaca 13 judul *slide*, argumennya tetap utuh, yaitu tesis dan tiga
rekomendasi, dua fitur dan buktinya, arsitektur dan hasilnya, peta dan tiga tindakan, lalu batas dan ajakan.

### B2. Pembagian waktu per bagian

| Bagian | *Slide* | Rentang | Durasi | Pembicara |
|---|---|---|---|---|
| I. Pembukaan dan ringkasan eksekutif | 1, 2 | 00:00–01:00 | 1 menit | Orang 1 (jangkar) |
| II. Rekayasa fitur (bobot 25) | 3, 4, 5 | 01:00–03:30 | 2 menit 30 detik | Orang 2 (teknis) |
| III. Arsitektur model (bobot 15) | 6, 7 | 03:30–05:00 | 1 menit 30 detik | Orang 2 (teknis) |
| IV. Rekomendasi kebijakan publik (bobot 35) | 8, 9, 10, 11 | 05:00–08:30 | 3 menit 30 detik | Orang 3 (kebijakan) |
| V. Keterbatasan dan penutup | 12, 13 | 08:30–09:30 | 1 menit | Orang 1 (jangkar) |
| *Buffer* (di luar narasi) | | 09:30–10:00 | 30 detik | serah terima dan jeda |

Pembagian pembicara mengikuti kartu latihan *deck* v2 (jangkar, teknis, kebijakan). Tiga kali serah terima sudah
diucapkan di akhir *slide* 2, 7, dan 11 dengan kalimat tanpa nama ("rekan kami ..."), sehingga catatan pembicara tetap
benar siapa pun yang tampil. Tim boleh menyebut nama pembicara berikutnya saat serah terima. Jedanya masuk ke
*buffer*. Bila latihan di Zoom menunjukkan serah terima lebih lama dari sekitar 8 detik, operator *slide* langsung
berpindah tanpa kalimat serah terima.

### B3. Durasi dan kepadatan narasi per *slide*

Kecepatan diukur dalam kata lisan: angka dihitung sesuai cara pengucapannya, misalnya 0,24844 menjadi tujuh kata
("nol koma dua empat delapan empat empat"). Rentang yang diterima 120 sampai 140 kata per menit.

<!-- tabel-waktu:mulai -->
| *Slide* | Rentang | Durasi | Kata tertulis | Kata lisan | Kata per menit |
|---|---|---|---|---|---|
| 1 | 00:00–00:25 | 25 detik | 50 | 52 | 125 |
| 2 | 00:25–01:00 | 35 detik | 73 | 80 | 137 |
| 3 | 01:00–01:55 | 55 detik | 120 | 123 | 134 |
| 4 | 01:55–02:45 | 50 detik | 98 | 104 | 125 |
| 5 | 02:45–03:30 | 45 detik | 98 | 101 | 135 |
| 6 | 03:30–04:30 | 60 detik | 128 | 138 | 138 |
| 7 | 04:30–05:00 | 30 detik | 58 | 69 | 138 |
| 8 | 05:00–05:55 | 55 detik | 107 | 117 | 128 |
| 9 | 05:55–06:45 | 50 detik | 93 | 105 | 126 |
| 10 | 06:45–07:35 | 50 detik | 95 | 111 | 133 |
| 11 | 07:35–08:30 | 55 detik | 116 | 127 | 139 |
| 12 | 08:30–09:05 | 35 detik | 76 | 77 | 132 |
| 13 | 09:05–09:30 | 25 detik | 56 | 56 | 134 |
| **Jumlah** | 00:00–09:30 | **570 detik** | 1168 | 1260 | 133 |
<!-- tabel-waktu:selesai -->

### B4. Sistem desain (melanjutkan *deck* v2 dan v3)

- **Format:** 16:9, huruf Calibri, judul 28 pt, isi 14 sampai 18 pt. Semua label, keterangan, dan teks tabel minimal
  14 pt. Hanya baris kaki (nama tim, nomor sel *notebook*, nomor *slide*) yang 12 pt.
- **Kepadatan:** rata-rata teks di layar paling banyak 100 kata per *slide* (judul, isi, label, dan tabel dihitung;
  catatan pembicara tidak).
- **Baris kaki *slide* utama:** penanda *notebook* asal angka di *slide* itu, ditulis seperti yang terlihat juri saat
  membuka *notebook*: nomor bagian untuk sel markdown dan nomor eksekusi In [n] untuk sel kode, misalnya "Notebook:
  bagian 7.2, In [35]". Dokumen ini dan baris Sumber di catatan memakai indeks sel dari 0, yang ikut menghitung sel
  markdown. Padanannya untuk sel kode yang dirujuk:

  | Indeks sel | 8 | 19 | 23 | 25 | 31 | 35 | 37 | 41 | 43 | 45 | 47 | 49 | 50 | 54 | 55 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Nomor eksekusi | In [6] | In [17] | In [21] | In [22] | In [25] | In [27] | In [28] | In [30] | In [31] | In [32] | In [33] | In [34] | In [35] | In [36] | In [37] |

  Sel markdown yang dirujuk: 24 (bagian 1), 28 (bagian 2.1 dan 2.2), 30 (bagian 3), 39 (bagian 6), 40 (bagian 7), 48
  (bagian 7.1 dan 7.2), dan 51 (bagian 7.3).
- **Warna:** biru tua `#16324F` untuk struktur; jingga `#D97B29` untuk gas dan angka terpenting; hijau kebiruan
  `#1F7A8C` untuk listrik dan hasil validasi; abu-abu untuk jalur tanpa model besar.
- **Latar gelap:** hanya *slide* 1 dan 13.
- **Label jenis bukti di layar:** FAKTA, HASIL, ASUMSI, dan INTERPRETASI, seperti *deck* v2. *Slide* kebijakan memakai
  rantai TEMUAN, INTERPRETASI, REKOMENDASI, dan LANGKAH.

## C. *Slide* utama (13 *slide*, 9 menit 30 detik)

### Slide 1 · 00:00–00:25 (25 detik)
- **Bagian rubrik:** I. Pembukaan dan ringkasan eksekutif
- **Judul Slide:** Trilema energi kota: elektrifikasi penuh hampir menggandakan konsumsi listrik zona
- **Visual Utama:** Latar gelap. Segitiga trilema di kiri dengan tiga sudut: emisi turun, jaringan listrik andal, dan
  biaya terjangkau. Di kanan, satu angka besar berwarna jingga, +94,85%, dengan label ASUMSI kecil di bawahnya. Tidak
  ada grafik model atau istilah teknis. Pesan yang harus ditangkap: ketiga tujuan saling tarik, dan tekanannya besar.
- **Isi Teks:**
  - Emisi turun · Jaringan listrik andal · Biaya terjangkau
  - +94,85%: median kenaikan konsumsi listrik tahunan zona bila semua pemanas gas diganti pompa panas
  - ASUMSI · 80% gas untuk pemanasan, COP 3 (3 kWh panas per 1 kWh listrik); energi tahunan, bukan beban
    puncak
- **Narasi Pembicara (±25 detik):** Bapak dan Ibu juri, kami tim BISMILLAH. Kota yang beralih dari pemanas gas ke
  pemanas listrik menghadapi trilema: emisi harus turun, jaringan listrik tetap andal, dan biaya energi tetap
  terjangkau. Dalam skenario kami, bila semua pemanas gas diganti pompa panas, konsumsi listrik tahunan zona naik
  dengan median hampir 95 persen.
- **Sumber:** *writeup* bab 1 (definisi trilema); sel 48 (asumsi skenario); sel 50 (median 94,85%). Baris kaki setiap
  *slide* memakai penanda yang terlihat di *notebook*: nomor bagian (judul markdown) dan nomor eksekusi In [n] sel kode;
  baris Sumber memakai indeks sel dari 0 (sel 50 = In [35]; peta lengkapnya di bagian B4 dokumen struktur)
- **Sel notebook:** bagian 7.2, In [35]

### Slide 2 · 00:25–01:00 (35 detik)
- **Bagian rubrik:** I. Pembukaan dan ringkasan eksekutif
- **Judul Slide:** Transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan
- **Visual Utama:** Tiga kartu bernomor sejajar, satu per rekomendasi, tanpa panah waktu yang menyiratkan satu urutan
  untuk semua zona. Setiap kartu memuat label tuntutan trilema yang dijawabnya (warna sudut segitiga *slide* 1) dan
  ikon peta kuadran 2 × 2 dengan kuadran sasarannya diwarnai, sama dengan kolom Program di tabel *slide* 8: R1 di Q1,
  R2 di Q1 dan Q3 (kenaikan listrik di atas median), R3 di Q2 (kenaikan listrik di bawah median). Kartu 2 berlatar
  gelap karena penilaian gardu menjadi syarat sebelum subsidi pompa panas dibuka. Kartu 1 memuat angka 118 zona dan
  47% gas dalam warna aksen. Satu baris kecil di bawah kartu menjadi penunjuk arah presentasi.
- **Isi Teks:**
  - 1 · *Retrofit* selubung dulu di 118 zona Q1 (47% konsumsi gas) · biaya dan emisi · Q1
  - 2 · Gardu dinilai sebelum subsidi pompa panas dibuka · keandalan jaringan · Q1 dan Q3: kenaikan listrik di atas
    median
  - 3 · Pilot pompa panas di kawasan bergas zona Q2 · transisi yang terukur · Q2: kenaikan listrik di bawah median
  - Berikutnya: fitur → model → kebijakan → batas
- **Narasi Pembicara (±35 detik):** Tesis kami: transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan.
  Pertama, untuk biaya dan emisi: *retrofit* selubung bangunan lebih dulu di 118 zona yang intensitas gas dan
  kenaikan listriknya sama-sama tinggi, yang memuat 47 persen konsumsi gas. Kedua, untuk keandalan: gardu dinilai
  sebelum subsidi pompa panas dibuka, terutama di zona yang kenaikan listriknya tinggi. Ketiga, agar transisi
  terukur: kawasan bergas dimulai dengan pilot pompa panas. Setelah ini, rekan kami menunjukkan dasar analisisnya.
- **Sumber:** sel 49 (118 zona, porsi gas 0,468); sel 51 (rekomendasi; butir 3: zona Q1 dan Q3 untuk gardu);
  *writeup* bab 1 (aspek trilema per target)
- **Sel notebook:** In [34], bagian 7.3

### Slide 3 · 01:00–01:55 (55 detik)
- **Bagian rubrik:** II. Rekayasa fitur (25)
- **Judul Slide:** Luas lantai per meter: fitur teratas model penjelas konsumsi, dihitung tanpa menyentuh target
- **Visual Utama:** Tiga kolom berlabel DIBANGUN, RISIKO DIKENDALIKAN, dan KONTRIBUSI. Kolom kiri: ilustrasi dua rumah,
  rumah kecil "±37 m² per meter" dan rumah besar "±173 m² per meter" (rata-rata 5% kawasan terkecil dan terbesar),
  dengan kartu rumus satu baris di atasnya: luas lantai (`bld_dim_06`) ÷ jumlah meter (`bld_dim_12`). Kolom tengah:
  empat tanda centang pengaman. Kolom kanan: angka besar 25–30%, yaitu porsi kontribusi SHAP kelompok rasio fisik
  turunan pada model penjelas EUI, gas per meter, dan listrik per meter (*writeup* B Tabel 3; grafik sel 43; dihitung
  ulang dari nilai SHAP oleh `make_visuals_v4.py`). Di bawahnya dua diagram batang lima fitur teratas model penjelas
  untuk intensitas gas dan listrik per meter (grafik asli PowerPoint dari `img/data_grafik.json`, nilai sel 45).
  Fitur turunan buatan tim (`area_per_meter`, `area_per_epc`, `s002_ratio`, `meters_per_epc`; kolom "keluarga" di
  sel 45) berbatang biru tua dengan `area_per_meter` jingga, sedangkan kolom asli data berbatang abu-abu. Legenda
  warna dan keterangan "rata-rata |SHAP|, sampel 20.000 baris latih" ada di bawah grafik.
- **Isi Teks:**
  - DIBANGUN · `area_per_meter` = luas lantai ÷ jumlah meter ≈ luas satu rumah
  - RISIKO DIKENDALIKAN · dua kolom input baris yang sama · tanpa target · tanpa statistik lintas baris · rumus sama
    di data latih dan uji
  - KONTRIBUSI · 25–30%: porsi SHAP rasio fisik turunan · peringkat 1: 2,8× fitur kedua (EUI) dan 3,0× (listrik)
  - fitur turunan buatan tim · kolom asli data · rata-rata |SHAP|
- **Narasi Pembicara (±55 detik):** Dasar pertama adalah fitur. Fitur fisik utama kami sederhana: luas lantai kawasan
  dibagi jumlah meteran. Satu meteran kira-kira satu rumah tangga, jadi angka ini kira-kira luas rata-rata satu
  rumah. Fitur yang diam-diam membawa jawaban membuat skor tampak bagus, tetapi gagal di data baru. Risiko itu kami
  kendalikan sejak desain: fitur ini hanya membagi dua kolom input pada baris yang sama, tanpa target, dengan rumus
  yang sama untuk data latih dan data uji. Kontribusinya terlihat di model penjelas, yaitu model terpisah yang hanya
  memakai karakteristik terukur dan dibaca dengan SHAP. Kelompok rasio fisik turunan menyumbang 25 sampai 30 persen,
  dan fitur ini peringkat pertama untuk intensitas gas dan listrik per rumah, sekitar tiga kali fitur kedua. Inilah
  dasar rekomendasi *retrofit* nanti.
- **Sumber:** sel 28 (rumus dan alasan bebas target); sel 40 (model penjelas); sel 43 dan *writeup* B Tabel 3 (porsi
  rasio fisik 25%, 30%, 26%); sel 45 (mean |SHAP| 0,0895; 0,0321; 0,0294; keluarga "rasio fisik turunan");
  `deck/ppt01/bukti/cek_kebijakan.json` (±37 dan ±173 m²)
- **Sel notebook:** bagian 2.1, In [31], [32]

### Slide 4 · 01:55–02:45 (50 detik)
- **Bagian rubrik:** II. Rekayasa fitur (25)
- **Judul Slide:** Fitur kawasan termirip memakai target tetangga, tetapi dihitung *out-of-fold* dua lapis
- **Visual Utama:** Bagian atas: pita nomor subzona dengan satu kawasan di tengah dan tiga jendela bertingkat (±30,
  ±100, ±300), diberi catatan "96,5% pasangan subzona bernomor berurutan berada di zona yang sama". Bagian bawah:
  skema *cross-fit* dua lapis, bentuknya sama dengan *slide* 5 *deck* v2. Baris pertama berisi sepuluh kotak *fold*:
  satu disorot "dinilai", sembilan lainnya diberi satu kurung berlabel "9 *fold* lain: sumber tetangga". Baris kedua
  memecah bagian latih menjadi lima: satu disorot "dihitung", empat lainnya diberi kurung "4 bagian lain: sumber
  tetangga". Panah ketiga menunjukkan data uji mengambil tetangga dari bagian latih model yang sama.
- **Isi Teks:**
  - DIBANGUN · rata-rata konsumsi k kawasan paling mirip, dalam jendela nomor subzona
  - 96,5% pasangan subzona bernomor berurutan berada di zona yang sama → nomor berdekatan = pendekatan lokasi
  - RISIKO DIKENDALIKAN · lapis luar 10 *fold*, lapis dalam 5 bagian: tidak ada baris memakai targetnya sendiri
- **Narasi Pembicara (±50 detik):** Fitur kedua bersifat spasial. Untuk setiap kawasan, kami mencari kawasan dengan
  stok bangunan paling mirip di sekitarnya, lalu memakai rata-rata konsumsinya. Sekitar berarti nomor subzona yang
  berdekatan: 96,5 persen pasangan subzona bernomor berurutan berada di zona yang sama, jadi penomorannya kami pakai
  sebagai pendekatan kedekatan lokasi. Karena fitur ini memakai target tetangga, risikonya adalah kebocoran target.
  Maka fitur ini dihitung *out-of-fold* dua lapis. Lapis luar: baris yang sedang dinilai hanya melihat tetangga dari
  sembilan *fold* lain. Lapis dalam: baris latih pun hanya melihat tetangga dari empat bagian lain, supaya model tidak
  belajar dari fitur yang memuat targetnya sendiri.
- **Sumber:** sel 28 (definisi dan skema dua lapis); `train_lib.Data.KNN_CONFIGS` (sel 8: jendela 30, 100, 300);
  `deck/ppt02/bukti/cek_zona_contoh.json` (96,5%)
- **Sel notebook:** bagian 2.2, In [6]

### Slide 5 · 02:45–03:30 (45 detik)
- **Bagian rubrik:** II. Rekayasa fitur (25)
- **Judul Slide:** Uji label acak lulus: label baris yang dinilai tidak terbaca, versi bocor tertangkap
- **Visual Utama:** Panel uji tiga baris dengan ikon hasil. Baris 1 (hijau): label validasi diacak, fitur validasi dan
  uji tidak berubah. Baris 2 (hijau): label satu bagian dalam baris latih diacak, fitur bagian itu identik, sedangkan
  bagian lain berubah sesuai rancangan. Baris 3 (merah): kontrol yang sengaja bocor berubah. Di bawahnya, strip
  KONTRIBUSI dengan panah kecil ke ikon peta kuadran. Label cakupan di bawah panel uji.
- **Isi Teks:**
  - Label 42.931 baris validasi diacak → fitur validasi dan 183.987 kawasan uji: 0 baris berubah
  - Label 77.275 baris latih bagian dalam diacak → fitur bagian itu identik; 309.097 baris lain berubah, sesuai
    rancangan
  - KONTROL · versi bocor berubah di 42.931 baris
  - KONTRIBUSI · galat EUI model tunggal −0,0022; prediksi untuk 30% nilai peta zona memakai fitur ini
  - Cakupan: dua blok fitur, satu *fold*
- **Narasi Pembicara (±45 detik):** Desain itu kami uji, bukan kami asumsikan. Label target konsumsi pada lebih dari 42
  ribu baris validasi kami acak, lalu fitur dihitung ulang. Fitur validasi dan kawasan uji tetap identik sampai bit
  terakhir. Untuk baris latih, label satu bagian dalam diacak: fitur bagian itu tetap identik, sedangkan bagian lain
  berubah karena memakainya sebagai tetangga, sesuai rancangan. Versi yang sengaja dibuat bocor langsung berubah, jadi
  uji ini mampu menangkap kebocoran. Cakupannya dua blok fitur pada satu *fold*. Fitur ini memberi perbaikan galat EUI
  terbesar dari satu kelompok fitur, dan prediksi kawasan uji, sekitar 30 persen nilai peta zona, memakainya.
- **Sumber:** `deck/ppt01/bukti/uji_kebocoran_notebook.json` dan `uji_kebocoran_oof_dalam.json`; sel 28 (−0,0022);
  sel 24 (data uji tepat 30% dari seluruh baris)
- **Sel notebook:** bagian 1, 2.2 · uji label acak: skrip terpisah

### Slide 6 · 03:30–04:30 (60 detik)
- **Bagian rubrik:** III. Arsitektur model (15)
- **Judul Slide:** Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik
- **Visual Utama:** Judul diambil dari *slide* 6 *deck* v2 karena menonjolkan dekomposisi bersih: tiga fakta struktur
  data menentukan siapa yang butuh model besar. Diagram alur tiga jalur mendatar yang dibedakan warna. Kotak kiri
  "Data latih + uji, 201 kolom" bercabang ke tiga jalur:
  - Jalur 1 (jingga): Gas dan EUI, 28 model (XGBoost, LightGBM, MLP, NCA).
  - Jalur 2 (hijau kebiruan): Listrik, 13 model.
  - Kurung kurawal membungkus kotak model jalur 1 dan 2 dengan label "41 model di-*blend*". Dari kedua jalur ini, panah
    menuju tiga tahap berwarna biru tua: gabungan bobot adaptif per jalur, koreksi galat tetangga, dan penyesuaian
    akhir.
  - Jalur 3 (abu-abu) berlabel "Porsi bersih dan adopsi: tanpa model besar", karena jalur ini memuat dua target:
    porsi bangunan bersih dari rumus eksak, adopsi disalin dari nilai subzona (91,86% kawasan uji), dan 2 LightGBM
    khusus untuk 8,14% kawasan uji di subzona baru. Garis jalur 3 sengaja tidak melewati kotak gabungan, kotak 2
    LightGBM bergaris putus-putus merah dengan label "terpisah", dan penjumlahan "43 model = 41 di-*blend* + 2
    terpisah" ditulis di bawah kolom keluaran.
  - Kolom kanan memuat galat jujur gabungan sebelum koreksi galat tetangga dan penyesuaian akhir (keluaran sel 37),
    sejajar dengan lima keluaran (gas, EUI, listrik, porsi bersih, adopsi), dengan skala "menebak rata-rata ≈ 1".
  - Pita bawah: tiga fakta struktur data dari *slide* 3 *deck* v2 (sel 25), masing-masing di bawah jalur yang
    ditentukannya: gas = EUI × luas lantai (jalur 1), porsi bersih = rumus, dan adopsi seragam per subzona (jalur 3).
- **Isi Teks:**
  - Jalur 1 · Gas dan EUI: 28 model
  - Jalur 2 · Listrik: 13 model
  - Jalur 1 dan 2: gabungan bobot adaptif → koreksi galat tetangga → penyesuaian akhir
  - Jalur 3 · Porsi bersih dan adopsi: tanpa model besar · porsi bersih: rumus · adopsi: salin subzona (91,86%) · 2
    LightGBM terpisah (8,14% kawasan uji di subzona baru)
  - 43 model = 41 di-*blend* + 2 terpisah
  - Galat jujur sebelum koreksi (RMSE/σ; menebak rata-rata ≈ 1): 0,2449 · 0,6502 · 0,3159 · 0 · ≈ 0,04
  - FAKTA DATA · Gas = EUI × luas lantai: galat relatif maks 6,7 × 10^−16^ · Porsi bersih = rumus: tepat di 100%
    baris latih · Adopsi seragam per subzona: 100% subzona latih
- **Narasi Pembicara (±60 detik):** Arsitekturnya mengikuti struktur data. Tiga fakta di bawah kami temukan sebelum
  model dilatih: gas selalu sama dengan EUI dikali luas lantai, porsi bangunan bersih tepat sama dengan satu rumus,
  dan adopsi seragam dalam satu subzona. Karena metrik menilai setiap kolom terpisah, setiap target boleh punya
  jalur sendiri. Jalur pertama untuk gas dan EUI dengan 28 model, karena keduanya satu besaran. Jalur kedua untuk
  listrik dengan 13 model. Di dua jalur ini, model digabung dengan bobot adaptif, lalu dikoreksi dengan galat
  tetangga dan disesuaikan dengan batas data; angka di kanan adalah galat jujur sebelum koreksi. Jalur ketiga tanpa
  model besar: porsi bersih dihitung dengan rumus, dan adopsi disalin dari subzonanya, kecuali 8,14 persen kawasan
  uji di subzona baru. Jadi totalnya 43 model: 41 di-*blend*, 2 LightGBM berdiri sendiri.
- **Sumber:** sel 23 (28 dan 13); sel 24 dan 25 (identitas: galat relatif maksimum 6,7e-16, 100,00% baris latih,
  100,00% subzona, 91,86%); sel 30 (tiga jalur); sel 35 (dua LightGBM adopsi, porsi 0,0814; galat jujur); sel 39
  (ringkasan galat); *writeup* bab 1 (skala metrik); `deck/ppt01/pratinjau/slide-03.png` dan `slide-06.png` (asal
  pita fakta dan judul)
- **Sel notebook:** bagian 1 dan 3, In [21], [22], [27]

### Slide 7 · 04:30–05:00 (30 detik)
- **Bagian rubrik:** III. Arsitektur model (15)
- **Judul Slide:** Skor publik turun di setiap langkah, dan skor privat dari data tersembunyi tidak memburuk
- **Visual Utama:** Kiri, dua pertiga lebar: grafik garis asli PowerPoint skor *leaderboard* publik per langkah
  pengembangan, sama dengan grafik *slide* 7 *deck* v2 (tujuh titik dari 0,25025 ke 0,24913, keluaran sel 37), dengan
  titik jingga skor privat 0,24844 di langkah terakhir. Label langkah memakai nama pendek tanpa istilah teknis, dan
  nilai setiap titik ditulis sebagai kotak teks agar koma desimal tetap. Kanan: dua ubin angka, 0,24844 (skor privat,
  peringkat 8, publik 0,24913) dan 0,4997 (AUC adversarial). Kartu sidik sha256 *deck* v3 diganti grafik ini; sidiknya
  pindah ke baris kaki bersama nomor sel.
- **Isi Teks:**
  - Skor *leaderboard* publik per langkah (makin kecil makin baik) · privat 0,24844 · peringkat 8
  - 0,24844 · skor privat, peringkat 8 · publik 0,24913: tidak memburuk
  - 0,4997 · AUC adversarial: latih dan uji tak dapat dibedakan, jadi validasi meniru data uji
- **Narasi Pembicara (±30 detik):** Apakah arsitektur ini bekerja di data baru? Skor publik turun di setiap langkah
  pengembangan. Skor privat dari data yang tidak pernah kami lihat adalah 0,24844, peringkat 8, tidak memburuk.
  Validasi kami meniru kondisi uji, karena data latih dan uji tidak dapat dibedakan: AUC 0,4997. Berkas submisinya
  identik setiap diulang di perangkat yang sama. Rekan kami melanjutkan ke kebijakan.
- **Sumber:** Kaggle LB privat (24 Okt 2026; `reports/STATUS.md`); sel 35 (sha256); sel 37 (LB publik per langkah:
  0,25025; 0,24954; 0,24943; 0,24934; 0,24925; 0,24920; 0,24913); sel 55 (AUC); `deck/ppt01/pratinjau/slide-07.png`
  (asal grafik)
- **Sel notebook:** In [27] sha256 aec745ab…d66e, [28], [37]

### Slide 8 · 05:00–05:55 (55 detik)
- **Bagian rubrik:** IV. Rekomendasi kebijakan publik (35)
- **Judul Slide:** Peta 330 zona: 118 zona Q1 memuat 47% konsumsi gas, jadi program diurutkan per kuadran
- **Visual Utama:** Peta kuadran 330 zona (`img/peta_kuadran.png`, digambar dari data zona sel 49). Sumbu
  mendatar menunjukkan intensitas gas zona (kWh/m²), sumbu tegak menunjukkan kenaikan konsumsi listrik tahunan (%),
  garis putus-putus menandai median kedua sumbu, dan ukuran titik menunjukkan konsumsi gas zona. Q1 berwarna jingga
  dan setiap kuadran diberi label programnya. Di kanan: tabel kuadran empat baris, kotak ASUMSI, dan kotak UJI
  KETAHANAN. Kolom Program sama dengan kartu *slide* 2: R2 (gardu) di Q1 dan Q3. Catatan kecil di bawah peta: bukan
  peta geografis karena data tanpa koordinat, dan 30% nilai zona berasal dari prediksi kawasan uji (*writeup* 7.3).
- **Isi Teks:**

  | Kuadran | Zona | Porsi gas | Indikator *retrofit* | Kenaikan listrik | Program |
  |---|---|---|---|---|---|
  | Q1 | 118 | 47% | 3,65 TWh | +102% | R1 + R2 |
  | Q2 | 47 | 12% | 1,07 TWh | +90% | R3 |
  | Q3 | 47 | 15% | 0,86 TWh | +99% | R2 |
  | Q4 | 118 | 27% | 1,81 TWh | +87% | standar |

  - Bukan peta geografis (tanpa koordinat); 30% nilai zona dari prediksi kawasan uji
  - ASUMSI · 80% gas untuk pemanasan, COP 3, energi tahunan · *retrofit* 7,39 TWh = indikator, bukan janji
  - UJI KETAHANAN · tahan derau acak per kawasan (Spearman 0,998); peka COP per iklim (Spearman 0,81) → daftar
    kandidat
- **Narasi Pembicara (±55 detik):** Sekarang kebijakan. Nilai aktual kawasan latih dan prediksi kawasan uji, total
  613 ribu kawasan, kami ringkas menjadi 330 zona. Ini peta kuadran, bukan peta geografis. Sumbu mendatar adalah
  intensitas gas zona; sumbu tegak adalah kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas.
  Garis median membagi empat kuadran. Kuadran satu berisi 118 zona yang berada di atas median pada kedua sumbu, dan
  memuat 47 persen konsumsi gas. Penilaian gardu menyasar kuadran satu dan tiga, yang kenaikan listriknya di atas
  median. Uji ketahanan menunjukkan urutan zona tahan terhadap derau prediksi acak, tetapi berubah bila efisiensi
  pompa panas berbeda per iklim. Karena itu peta ini adalah daftar kandidat.
- **Sumber:** sel 48 (definisi), 49 (tabel kuadran), 50 (141,35 TWh; 7,39 TWh), 51 butir 3 (gardu di Q1 dan Q3);
  `deck/ppt01/bukti/cek_kebijakan.json` (Spearman); `explain.zone_indices` (sel 19, ambang median)
- **Sel notebook:** bagian 7.2 dan 7.3, In [34], [35]

### Slide 9 · 05:55–06:45 (50 detik)
- **Bagian rubrik:** IV. Rekomendasi kebijakan publik (35)
- **Judul Slide:** Rekomendasi 1: *retrofit* selubung lebih dulu di zona Q1, mulai dari audit energi
- **Visual Utama:** Kiri: panel dependensi SHAP luas per meter terhadap gas per m² (`img/dep_gas_ukuran.png`),
  kedua ujungnya ditandai "±37 m²" dan "±173 m²", dengan sketsa kecil rumah kecil dan rumah besar. Di bawahnya:
  batang indikator potensi tiga zona kandidat (grafik asli PowerPoint dari `img/data_grafik.json`). Kanan: rantai
  empat kotak TEMUAN → INTERPRETASI → REKOMENDASI → LANGKAH, ditutup baris PELAKSANA dengan ikon orang.
- **Isi Teks:**
  - TEMUAN · rumah terkecil: sumbangan gas per m² sekitar +65% vs terbesar (antarkuartil +34%); periode bld_era_15:
    +7% EUI
  - INTERPRETASI · dinding dan atap relatif luas; asosiasi, bukan sebab-akibat
  - REKOMENDASI · dinding, atap, jendela dulu, pompa panas kemudian; Q1 = 47% gas, indikator potensi 3,65 TWh
  - LANGKAH · audit energi REG_273, REG_270, REG_045 → pilot dengan pembanding → ukur gas per m²
  - PELAKSANA · pemerintah kota dan penyedia jasa *retrofit*
- **Narasi Pembicara (±50 detik):** Rekomendasi pertama menyangkut *retrofit*. Temuan SHAP, yaitu sumbangan
  tiap variabel pada prediksi model penjelas: kawasan rumah terkecil punya sumbangan gas per meter persegi sekitar 65
  persen lebih tinggi daripada rumah terbesar, dan satu periode bangunan sekitar 7 persen. Ini asosiasi, bukan
  sebab-akibat; kemungkinan karena dinding dan atap rumah kecil relatif luas, bagian yang disasar perbaikan
  selubung. Temuan model mendukung rekomendasi ini: *retrofit* selubung lebih dulu di zona kuadran satu. Karena
  potensinya baru indikator, langkah pertamanya audit energi di REG_273, REG_270, dan REG_045, lalu pilot dengan
  pembanding. Pelaksananya pemerintah kota bersama penyedia jasa *retrofit*.
- **Sumber:** sel 47 dan 48 (dependensi SHAP, bld_era_15, penjelasan selubung); sel 49 dan 50 (Q1, kandidat);
  `deck/ppt01/bukti/cek_kebijakan.json` (kontras 5% dan kuartil); `deck/ppt02/bukti/cek_zona_contoh.json`
  (kandidat tetap Q1); *slide* cadangan C6 (pelaksana)
- **Sel notebook:** bagian 7.1, In [33], [34], [35]

### Slide 10 · 06:45–07:35 (50 detik)
- **Bagian rubrik:** IV. Rekomendasi kebijakan publik (35)
- **Judul Slide:** Rekomendasi 2: kapasitas gardu di zona Q1 dan Q3 dinilai sebelum subsidi pompa panas dibuka
- **Visual Utama:** Kiri atas: dasar SHAP rekomendasi ini, yaitu dependensi SHAP luas lantai per meter terhadap
  listrik per meter pada model penjelas (`img/dep_listrik_ukuran.png`, panel "listrik per rumah" dari *slide* 8 *deck*
  v2 yang digambar ulang dengan sumbu yang benar), dengan label "rumah besar vs kecil: sumbangan SHAP +75%" dan titik rata-rata
  pita 5% terkecil (±37 m²) dan terbesar (±173 m²). Kiri bawah: batang lima zona dengan kenaikan konsumsi listrik
  tahunan tertinggi dan batang median 330 zona (grafik asli PowerPoint dari `img/data_grafik.json`), dengan peringkat
  setiap zona pada skenario COP per iklim di kanan: REG_290 turun ke peringkat 47, sedangkan REG_270 naik ke peringkat
  1 dan diberi warna aksen sebagai titik awal penilaian. Kanan: rantai TEMUAN → INTERPRETASI → REKOMENDASI → LANGKAH,
  ditutup baris PELAKSANA.
- **Isi Teks:**
  - TEMUAN · SHAP: sumbangan listrik per rumah +75% di rumah besar vs kecil; lima zona tertinggi +113% sampai +121%
  - INTERPRETASI · konsumsi dasar dan tambahannya berbeda antarzona; energi tahunan, bukan beban puncak
  - REKOMENDASI · gardu dinilai per zona sebelum subsidi; daftar ini skrining, bukan bukti kapasitas
  - LANGKAH · operator menilai mulai REG_263, REG_270, REG_045 → penguatan bila perlu → subsidi bertahap
  - PELAKSANA · operator jaringan distribusi dan pemerintah kota
- **Narasi Pembicara (±50 detik):** Rekomendasi kedua menyangkut keandalan jaringan. Temuan SHAP: ukuran rumah adalah
  fitur teratas untuk listrik per rumah, dan sumbangannya di kawasan rumah besar sekitar 75 persen lebih tinggi. Jadi
  konsumsi listrik dasar sudah berbeda jauh antarkawasan. Skenario elektrifikasi menambah 113 sampai 121 persen di lima zona
  tertinggi. Ini energi tahunan; beban puncak dan kapasitas gardu tidak ada di data. Temuan model mendukung
  rekomendasi ini sebagai daftar skrining: gardu dinilai per zona sebelum subsidi, mulai REG_270 dan REG_045, yang
  tetap sepuluh besar pada kedua skenario COP, serta REG_263 dengan tambahan terbesar. Pelaksananya operator jaringan
  dan pemerintah kota.
- **Sumber:** sel 45 (`area_per_meter` peringkat 1 listrik per meter); sel 47 dan 48 butir 1 (rumah besar sekitar 75%
  lebih banyak listrik per rumah); sel 48 (energi tahunan, data tanpa profil beban); sel 50 (sepuluh zona kenaikan
  tertinggi); sel 51 butir 3 (Q1 dan Q3); `deck/ppt01/bukti/cek_kebijakan.json` (74,6%; tambahan absolut);
  `deck/ppt02/bukti/cek_zona_contoh.json` (peringkat pada kedua skenario)
- **Sel notebook:** bagian 7.1 dan 7.3, In [32], [33], [35]

### Slide 11 · 07:35–08:30 (55 detik)
- **Bagian rubrik:** IV. Rekomendasi kebijakan publik (35)
- **Judul Slide:** Rekomendasi 3: elektrifikasi kawasan bergas dimulai sebagai pilot pompa panas di zona Q2
- **Visual Utama:** Kiri, visual utama selebar setengah *slide*: dependensi SHAP adopsi terhadap porsi pemanas gas
  jaringan di kawasan dari model penjelas (`img/dep_adopsi_besar.png`, versi besar grafik *slide* 12 *deck* v2 yang
  digambar ulang), dengan anotasi "di bawah 40%: +4 sampai +12 poin" dan "di atas 80%: sumbangan negatif" (sel 48
  butir 4) serta pita jingga di atas 80%. Diagram batang tingkat adopsi *deck* v3 tidak lagi ditampilkan; angkanya
  tetap di kotak TEMUAN. Kanan: rantai TEMUAN → INTERPRETASI → REKOMENDASI → LANGKAH, ditutup baris PELAKSANA.
  Kotak REKOMENDASI memuat anotasi zona contoh yang tetap Q2 pada kedua skenario COP, REG_033: gas jaringan 83,21%,
  adopsi 4,99%. Kotak LANGKAH memuat alur pilot: cek gardu lokal, pilot dan audit energi dengan zona pembanding, ukur
  adopsi, lalu perluas bila terukur.
- **Isi Teks:**
  - TEMUAN · sumbangan SHAP adopsi turun saat gas jaringan naik; tingkat adopsi 40,24% (gas jaringan ≤40%) vs 2,73%
    (>80%, 70% kawasan)
  - INTERPRETASI · elektrifikasi kawasan bergas tertinggal; pola komposisi, bukan sebab-akibat
  - REKOMENDASI · pilot pompa panas + bantuan bongkar ketel lama di zona Q2, misalnya REG_033
  - LANGKAH · cek gardu lokal → pilot dan audit energi dengan pembanding → ukur adopsi → perluas bila terukur
  - PELAKSANA · pemerintah kota dan penyedia pompa panas
- **Narasi Pembicara (±55 detik):** Rekomendasi ketiga menyangkut adopsi. Temuan SHAP: sumbangan porsi gas
  jaringan pada adopsi turun dari positif menjadi negatif di atas 80 persen. Tingkat adopsi sekitar 40 persen di
  kawasan yang sedikit tersambung gas, tetapi hanya 2,73 persen di kawasan bergas di atas 80 persen, padahal kawasan
  seperti itu 70 persen dari seluruh kawasan. Karena porsi jenis pemanas berjumlah 100 persen, ini pola komposisi, bukan
  sebab-akibat. Temuan model mendukung rekomendasi ini: kawasan bergas perlu dorongan, dimulai dengan pilot pompa
  panas di zona kuadran dua seperti REG_033, yang kenaikan listriknya di bawah median. Pilot didahului cek gardu,
  disertai audit energi, memakai zona pembanding, dan diperluas hanya bila terukur. Pelaksananya pemerintah kota dan penyedia pompa panas.
  Rekan kami akan menutup.
- **Sumber:** `deck/ppt01/bukti/cek_kebijakan.json` (tingkat adopsi per porsi gas jaringan); sel 47 dan 48 butir 4
  (pola SHAP, +4 sampai +12 poin di bawah 40%, negatif di atas 80%); sel 51 butir 2 dan 4;
  `deck/ppt02/bukti/cek_zona_contoh.json` (REG_033 tetap Q2); `deck/ppt01/pratinjau/slide-12.png` (asal grafik)
- **Sel notebook:** bagian 7.1 dan 7.3, In [33]

### Slide 12 · 08:30–09:05 (35 detik)
- **Bagian rubrik:** V. Keterbatasan dan penutup
- **Judul Slide:** Rekomendasi ini dasar prioritas, bukan pengganti audit teknis atau validasi lapangan
- **Visual Utama:** Tiga kartu berjajar dengan pola "BELUM DIKETAHUI → PENANGANAN". Kartu 1: beban puncak dan
  kapasitas gardu terukur → penilaian operator. Kartu 2: COP lokal, ambang median, dan makna kolom yang disamarkan →
  daftar zona sebagai kandidat. Kartu 3: sebab-akibat dan potensi *retrofit* nyata → audit energi dan pilot dengan
  zona pembanding. *Slide* ini tidak memuat angka.
- **Isi Teks:**
  - Beban puncak dan kapasitas gardu terukur: tidak ada di data → penilaian operator jaringan
  - COP lokal, ambang median, dan makna kolom: belum pasti → daftar zona = kandidat
  - Efek sebab-akibat dan penghematan nyata: belum terukur → audit energi dan pilot dengan zona pembanding
- **Narasi Pembicara (±35 detik):** Ada tiga batas yang kami akui. Pertama, data hanya memuat energi tahunan; beban
  puncak dan kapasitas gardu yang terukur tidak tersedia, jadi penilaian gardu tetap pekerjaan operator. Kedua,
  efisiensi pompa panas lokal belum diketahui, ambang kuadran hanya median, dan makna kolom yang disamarkan masih
  dugaan, jadi daftar zona adalah kandidat. Ketiga, SHAP bukan sebab-akibat, dan potensi *retrofit* baru indikator.
  Karena itu rekomendasi ini adalah dasar prioritas dan evaluasi lanjutan, bukan pengganti audit teknis atau
  validasi lapangan.
- **Sumber:** sel 28, 40, 48, dan 51 (keterbatasan); `deck/ppt01/bukti/cek_kebijakan.json` (kepekaan COP, komposisi
  potensi)
- **Sel notebook:** bagian 2, 7, 7.2, dan 7.3

### Slide 13 · 09:05–09:30 (25 detik)
- **Bagian rubrik:** V. Keterbatasan dan penutup
- **Judul Slide:** Diurutkan per kuadran: *retrofit* dulu di Q1, gardu dinilai di Q1 dan Q3, pilot pompa panas di Q2
- **Visual Utama:** Latar gelap. Tiga kartu yang sama dengan *slide* 2, masing-masing dengan kuadran sasaran dan
  tuntutan trilema yang dijawabnya, untuk menutup lingkaran argumen. Di bawah setiap kartu, tepat satu kolom dengannya,
  langkah berikutnya untuk rekomendasi itu: audit energi untuk R1, data beban dan gardu untuk R2, satu pilot untuk
  R3. *Slide* ini tidak memuat angka baru.
- **Isi Teks:**
  - 1 *Retrofit* di Q1: biaya dan emisi · 2 Gardu di Q1 dan Q3: keandalan jaringan · 3 Pilot di Q2: transisi
    terukur
  - Langkah berikutnya: audit energi di zona kandidat Q1 · data beban puncak dan gardu dari operator · satu pilot Q2
    dengan zona pembanding
  - Terima kasih
- **Narasi Pembicara (±25 detik):** Dengan kenaikan listrik hampir dua kali lipat, trilema ini tidak selesai dengan
  satu program untuk semua zona. *Retrofit* di kuadran satu menjawab biaya dan emisi, penilaian gardu di kuadran satu
  dan tiga menjaga keandalan, dan pilot di kuadran dua menguji transisinya. Langkah pertamanya: audit energi, data
  gardu dari operator, dan satu pilot dengan pembanding. Terima kasih.
- **Sumber:** rangkuman *slide* 1, 2, 8 sampai 12; sel 51 butir 1 (kebutuhan panas turun mengurangi tagihan dan emisi)
  dan butir 3 (gardu di Q1 dan Q3)
- **Sel notebook:** bagian 7.3

## D. *Slide* cadangan untuk tanya jawab (di luar 9 menit 30 detik)

Setiap *slide* cadangan memuat judul, visual atau bukti yang perlu ditampilkan, pertanyaan juri yang dijawabnya, dan
pedoman jawaban singkat. Jawaban rinci tetap tersedia di `deck/ppt01/06_qna_juri_v2.md` dan `deck/ppt00/03_qna_juri.md`;
keduanya memakai nomor *slide* *deck* v2, jadi rujukan *slide* di sana tidak sama dengan *deck* ini.

### C1. Bobot adaptif A2xM2xp2 tidak memburukkan kolom mana pun pada skor jujur
- **Visual atau bukti:** Diagram basis pojok (`img/basis_pojok.png`). Bobot setiap model berubah halus menurut
  peringkat luas (A), jumlah meter (M), dan prediksi gabungan global (p) di data latih, dengan 2 × 2 × 2 = 8 bobot
  pojok per model. Digabung dengan NNLS (*non-negative least squares*, kuadrat terkecil dengan bobot tak negatif)
  menjadi 224 bobot gas, 224 bobot EUI, dan 104 bobot listrik untuk 429.303 baris. Tabel skor jujur (bobot di-*fit*
  ulang di setiap *fold* luar):

  | Kolom | Global | Adaptif | Selisih |
  |---|---|---|---|
  | Gas | 0,24494 | 0,24480 | −0,000146 |
  | EUI | 0,65023 | 0,64978 | −0,000453 |
  | Listrik | 0,31587 | 0,31585 | −0,000027 |

- **Pertanyaan juri:** "Apa yang dimaksud bobot adaptif?" · "Apakah bobot sebanyak itu tidak *overfit*?" · "Berapa
  sumbangannya terhadap skor?"
- **Pedoman jawaban:** Model yang paling akurat di kawasan kecil belum tentu paling akurat di kawasan besar, sehingga
  bobotnya dibiarkan berubah halus menurut luas, jumlah meter, dan besar prediksi. Semua bobot tidak negatif dan
  jumlahnya kecil dibandingkan dengan 429.303 baris. Pada skor jujur, tidak ada kolom yang memburuk; perbaikan
  terbesar ada di EUI dan rata-rata lima target −0,000125. Catatan jujur: bentuk ini dipilih pada data yang sama,
  sehingga bukti independennya adalah *leaderboard* publik, yaitu 0,24934 menjadi 0,24925 (−0,00009). Tiga simpul
  tidak lebih baik daripada dua.
- **Sumber:** sel 23, 35, 37; `deck/ppt01/bukti/skor_jujur_bobot_adaptif.json`; `reports/DECISIONS.md` D012

### C2. Koreksi galat tetangga menggeser prediksi sebagian kecil mengikuti galat tetangga
- **Visual atau bukti:** Di *notebook*, tahap ini bernama koreksi galat wilayah. Diagram alir: galat *out-of-fold*
  gabungan → rata-rata galat per zona, petak iklim, ±30 nomor subzona, dan k kawasan termirip → dikecilkan n/(n + 30)
  atau n/(n + 5) bila tetangganya sedikit → dikali bobot λ yang dipilih di *holdout set* (30% data latih yang
  disisihkan untuk evaluasi internal) lalu dibekukan di `configs/resid_knn.json`. Tabel λ untuk gas (W = jendela
  nomor subzona, k = jumlah tetangga):

  | Fitur koreksi (gas) | λ |
  |---|---|
  | galat zona | 0,453 |
  | galat petak iklim | 0,346 |
  | galat ±30 subzona | −0,020 |
  | kNN full W100 k20 | 0,037 |
  | kNN era W30 k20 | 0,126 |
  | kNN fc W300 k50 | 0,079 |
  | kNN full W10 k10 | 0,072 |
  | kNN era W300 k50 | 0,124 |
  | kNN full W3000 k200 | 0,156 |

  Besar koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195 (sel 35). Koreksi tingkat grup sudah ada sejak v9;
  bagian kNN menurunkan *leaderboard* publik sekitar 0,0004 (0,2499 menjadi 0,24954). Angka 0,0007 di sel 39 (v12 ke
  v15) menggabungkan beberapa perubahan sekaligus (bagian E butir 13).
- **Pertanyaan juri:** "Apakah koreksi ini memakai target data uji?" · "Bagaimana λ dipilih, dan mengapa ada yang
  negatif?" · "Berapa besar pengaruhnya?"
- **Pedoman jawaban:** Setelah digabung, galat masih berkelompok: bila tetangga yang mirip sama-sama ditebak terlalu
  rendah, kawasan ini kemungkinan juga terlalu rendah. Untuk baris uji, yang dirata-rata hanya galat baris latih;
  target data uji tidak pernah dipakai. λ dipilih di *holdout set* yang tidak ikut menyusun tetangga, lalu
  dibekukan. λ negatif kecil muncul karena fitur tetangga saling berkorelasi; yang divalidasi adalah efek totalnya.
  Langkah "koreksi dan lainnya" di grafik *slide* 7 (0,25025 menjadi 0,24954) juga memuat model adopsi, MLP dua
  keluaran, salinan duplikat, dan pencabutan CatBoost; bagian kNN sendiri sekitar 0,0004.
- **Sumber:** sel 22, 35, 37, 39; `reports/DECISIONS.md` D015; `deck/fakta_turunan.md`; `reports/cv_lb_history.csv`

### C3. Dua blok fitur bertarget utama diuji dengan label acak pada satu *fold*
- **Visual atau bukti:** Tabel uji label acak (modul *notebook* yang sama, *fold* 0 dari 10). Yang diacak adalah
  kolom turunan tiga target konsumsi (gas, listrik, EUI) yang dipakai kedua blok fitur:

  | Blok fitur | Kolom | Baris yang diuji | Berubah |
  |---|---|---|---|
  | kNN-target, validasi | 39 | 42.931 | 0 |
  | kNN-target, data uji | 39 | 183.987 | 0 |
  | rata-rata wilayah, validasi | 42 | 42.931 | 0 |
  | rata-rata wilayah, data uji | 42 | 183.987 | 0 |
  | kNN-target, latih bagian 0 | 39 | 77.275 | 0; 309.097 baris lain berubah |
  | rata-rata wilayah, latih bagian 0 | 42 | 77.275 | 0; 309.097 baris lain berubah |
  | KONTROL: tetangga termasuk baris validasi | 15 | 42.931 | 42.931 |

  Pengendalian per kelompok fitur: rasio fisik seperti `area_per_meter` dihitung per baris tanpa target; agregat
  kawasan (jumlah kawasan, total luas, jumlah meter) dihitung dari fitur data latih dan uji tanpa target; rata-rata
  target wilayah dan kawasan termirip dihitung *out-of-fold* dua lapis. Di luar cakupan uji label acak: fitur
  tetangga model adopsi, koreksi galat tetangga, bobot gabungan, dan faktor penyesuaian akhir. Semuanya dirancang
  memakai prediksi *out-of-fold* atau *holdout set*, tetapi belum diuji dengan label acak.
- **Pertanyaan juri:** "Bagaimana Anda tahu tidak ada kebocoran?" · "Mengapa baris latih bagian lain berubah?" ·
  "Apakah agregat yang memakai data uji termasuk kebocoran?" · "Apakah *early stopping* membuat galat *out-of-fold*
  bias?"
- **Pedoman jawaban:** Baris validasi dan uji tidak membaca label validasi, dan baris latih tidak membaca label
  bagian dalamnya sendiri. Bagian lain berubah karena memakai bagian yang diacak sebagai tetangga; itu justru syarat
  lulus. Kontrol bocor membuktikan uji ini peka. Klaimnya terbatas pada dua blok fitur dan satu *fold*. Agregat
  kawasan hanya memakai kolom fitur, bukan target. XGBoost dan LightGBM memakai *early stopping* (penghentian
  pelatihan lebih awal) pada *fold* validasinya sendiri, sehingga galat *out-of-fold* sedikit optimistis; karena itu
  keputusan diambil dari *holdout set*. Bobot dari prediksi *out-of-fold* setara dengan bobot yang di-*fit* di
  *holdout set* (0,25096 vs 0,25095, D002). Yang diacak adalah kolom turunan tiga target konsumsi (gas, listrik, EUI), dengan modul
  *notebook* yang sama pada *fold* 0 dari 10.
- **Sumber:** `deck/ppt01/bukti/uji_kebocoran_notebook.py` dan `.json`, `uji_kebocoran_oof_dalam.json`; sel 28, 36;
  `reports/DECISIONS.md` D002; `adopt_v2.neighbour_feats` (sel 12)

### C4. Rincian 43 model: 28 jalur gas/EUI, 13 jalur listrik, dan 2 LightGBM adopsi
- **Visual atau bukti:** Tabel keluarga model per jalur (dihitung dari tabel anggota sel 31):

  | Keluarga | Gas/EUI | Listrik |
  |---|---|---|
  | XGBoost, termasuk varian Poisson | 16 | 3 |
  | LightGBM, termasuk varian Poisson | 5 | 4 |
  | MLP standar | 2 | 1 |
  | MLP PLR dan TabM-mini | 3 | 2 |
  | MLP gas + listrik | 1 | 2 |
  | NCA | 1 | 1 |
  | Jumlah | 28 | 13 |

  Jalur 3 memuat 2 LightGBM adopsi (`q_xent` dan `r_reg`) yang dirata-rata dan hanya dipakai untuk 8,14% kawasan
  uji; keduanya tidak ikut *blending*. Pita bawah: 43 = 41 + 2.
- **Pertanyaan juri:** "Mengapa 41 model, bukan satu model terbaik?" · "*Notebook* menulis 42 model; mana yang benar?"
  · "Apakah dua LightGBM adopsi ikut digabung?" · "Berapa model yang sebenarnya dilatih?"
- **Pedoman jawaban:** Angka 43 adalah jumlah model yang keluarannya masuk berkas submisi: 41 di dua kelompok gabungan
  berbobot adaptif, ditambah 2 LightGBM adopsi yang berdiri sendiri. Angka 41 dihitung per keluaran: MLP dua keluaran
  `mk2` adalah satu jaringan yang menyumbang satu anggota ke jalur gas/EUI dan satu ke jalur listrik. *Notebook*
  melatih 42 model di jalur 1 dan 2, yaitu 36 model dengan sepuluh *fold* dan 6 model dengan lima *fold*. Satu
  keluaran tidak terpilih, yaitu `mk1_gpm` (keluaran gas dari MLP dua keluaran yang keluaran listriknya terpilih).
  Setiap anggota dilatih per *fold*, sebagian dengan tiga *seed*, sehingga di balik 41 anggota terdapat 550 model
  terlatih. Gabungan memperbaiki model tunggal terbaik sebesar 0,0083 untuk EUI, 0,0036 untuk gas, dan 0,0027 untuk
  listrik; angka ini diukur pada *holdout set* versi v16, bukan pada gabungan final.
- **Sumber:** sel 23, 31, 32, 33, 35; *writeup* B bab 6; `reports/KAGGLE_WRITEUP_TOP20.md` bagian 3.3;
  `configs/members.yaml` (sel 21)

### C5. SHAP dibaca dari model penjelas terpisah dan hanya menunjukkan sumbangan pada prediksi
- **Visual atau bukti:** Tabel kecocokan empat model penjelas (R² pada data yang tidak dipakai untuk melatihnya): log
  EUI 0,563; log gas per meter 0,747; log listrik per meter 0,491; adopsi 0,991. Tabel porsi kontribusi per kelompok
  fitur (*writeup* B Tabel 3). Satu baris penegas: SHAP adalah sumbangan pada prediksi model penjelas, bukan efek
  sebab-akibat. Tabel "kalimat aman" *deck* v3 dipindah ke pedoman jawaban karena ditujukan bagi pembicara, bukan
  bagi juri.

- **Pertanyaan juri:** "Apakah SHAP membuktikan sebab-akibat?" · "Mengapa tidak membaca SHAP dari gabungan 41 model?"
  · "Mengapa R² adopsi 0,991?" · "Mengapa nilai 0,0895 muncul dua kali di sel 45?"
- **Pedoman jawaban:** Tidak; SHAP (*SHapley Additive exPlanations*) menguraikan prediksi model menjadi sumbangan
  aditif tiap fitur, bukan efek intervensi. Karena itu setiap program diusulkan sebagai pilot dengan zona pembanding.
  Gabungan 41 model memakai rata-rata target wilayah, yang bukan tuas kebijakan, sehingga dilatih satu LightGBM
  penjelas per target (800 putaran) yang hanya memakai karakteristik terukur. TreeSHAP eksak dihitung pada sampel
  20.000 baris latih untuk tabel dan grafik, sedangkan peta zona memakai kontribusi di seluruh 613 ribu kawasan. R²
  adopsi tinggi karena porsi jenis pemanas berjumlah 100%, bukan karena perilaku terjelaskan. Untuk alasan yang sama,
  porsi sosiodemografi 15% pada adopsi diukur terhadap total yang 81%-nya identitas komposisi, sehingga tidak dipakai
  sebagai bukti di *slide* utama. Peringkat `area_per_meter` adalah peringkat pada model penjelas, bukan ukuran
  sumbangannya pada akurasi prediksi. Nilai 0,0895 pada dua target adalah kebetulan pembulatan: nilai lengkapnya
  0,08953 dan 0,08948. Nama kolom disamarkan panitia, sehingga maknanya berupa dugaan, kecuali `bld_dim_06` sebagai
  luas lantai yang terbukti dari identitas EUI × luas lantai = gas. Kalimat aman: hindari "rumah kecil menyebabkan
  konsumsi gas tinggi", pakai "ukuran rumah yang kecil berkontribusi menaikkan prediksi gas per m² pada model
  penjelas"; hindari "sambungan gas menghambat adopsi", pakai "porsi gas jaringan yang tinggi berasosiasi dengan
  adopsi rendah, dan pola ini bersifat komposisi"; hindari "*retrofit* di Q1 menghemat 3,65 TWh", pakai "indikator
  potensi *retrofit* Q1 sebesar 3,65 TWh perlu diverifikasi dengan audit energi".
- **Sumber:** sel 40, 41, 42, 45; *writeup* B Tabel 3; `reports/policy/shap_eui.npz`, `shap_epm.npz`;
  `deck/ppt01/fakta_tambahan.md` bagian C dan E

### C6. Celah data dan rencana validasi lapangan
- **Visual atau bukti:** Dua blok.
  - Celah data: beban puncak (kW), kapasitas gardu, COP lokal, koordinat geografis, biaya *retrofit*, tarif, dan
    intensitas karbon listrik tidak tersedia. Makna kolom yang disamarkan berupa dugaan, dan ambang kuadran berupa
    median.
  - Rencana validasi:

  | Rekomendasi | Pelaksana | Validasi sebelum perluasan | Ukuran keberhasilan |
  |---|---|---|---|
  | R1 *retrofit* Q1 | pemerintah kota, penyedia jasa *retrofit* | audit energi REG_273, REG_270, REG_045 | gas per m² turun vs pembanding |
  | R2 gardu Q1 dan Q3 | operator jaringan, pemerintah kota | beban puncak dan kapasitas gardu REG_263, REG_270, REG_045 | tidak ada gardu melampaui kapasitas |
  | R3 pilot Q2 | pemerintah kota, penyedia pompa panas | cek gardu lokal, pilot bergiliran dan audit energi (REG_033) | adopsi naik vs pembanding |

- **Pertanyaan juri:** "Data apa yang masih dibutuhkan sebelum eksekusi?" · "Siapa yang menjalankan setiap
  rekomendasi?" · "Bagaimana memvalidasinya di lapangan?"
- **Pedoman jawaban:** Sebelum eksekusi dibutuhkan beban puncak dan kapasitas gardu dari operator, COP lokal per zona,
  dan hasil audit energi. Kenaikan yang dihitung adalah energi tahunan; beban puncak dapat jauh lebih besar karena
  pemanasan menumpuk di bulan terdingin dan COP turun saat dingin. Setiap program dijalankan sebagai pilot dengan
  zona pembanding sehingga dampaknya terukur sebelum diperluas. Subsidi R2 dibuka bertahap dengan tarif berbasis
  waktu pemakaian. Makna kolom yang disamarkan berupa dugaan, dan ambang kuadran berupa median.
- **Sumber:** sel 48, 51; `deck/ppt01/06_qna_juri_v2.md` P11

### C7. Validasi acak per baris meniru data uji, sedangkan validasi per zona akan membuang fitur yang berguna
- **Visual atau bukti:** Tabel *writeup* B Tabel 1 (LightGBM, RMSE/σ):

  | Skema | Rata-rata wilayah | Gas | EUI | Listrik |
  |---|---|---|---|---|
  | acak per baris | tanpa | 0,2573 | 0,6721 | 0,3262 |
  | acak per baris | dengan | 0,2562 | 0,6699 | 0,3255 |
  | per zona | tanpa | 0,2708 | 0,6923 | 0,3302 |
  | per zona | dengan | 0,3637 | 0,8331 | 0,3474 |

  Bukti mekanisme pemisahan data: AUC adversarial 0,4997; kawasan uji di subzona baru 8,14% vs harapan 8,17%. Bukti
  tidak *overfit* ke *leaderboard*: skor privat 0,24844 vs publik 0,24913; satu perubahan ditolak karena tidak
  dikonfirmasi *leaderboard* (0,24938 vs 0,24934, D011).
- **Pertanyaan juri:** "Mengapa tidak memakai validasi per wilayah (*GroupKFold*)?" · "Apakah model *overfit* ke
  *leaderboard* publik?"
- **Pedoman jawaban:** Data uji diambil acak per baris, sehingga 92% kawasan uji berada di subzona yang juga ada di
  data latih. Skema per zona menguji situasi yang tidak terjadi di data uji dan membuat rata-rata wilayah tampak
  merusak (EUI 0,6923 menjadi 0,8331). Komponen diterima berdasarkan *holdout set*; *leaderboard* hanya mengonfirmasi
  arah. Selisih privat dan publik yang kecil konsisten dengan tidak *overfit*, tetapi bukan bukti mutlak.
- **Sumber:** *writeup* B Tabel 1; sel 54, 55; Kaggle LB privat; `reports/DECISIONS.md` D011

### C8. Penyesuaian akhir menangani dua batas data, dan hasilnya identik setiap diulang di perangkat yang sama
- **Visual atau bukti:** Tabel penyesuaian:

  | Langkah | Gas | EUI | Listrik |
  |---|---|---|---|
  | S1: kawasan 2 sampai 3 meter | +20,1% | +18,4% | +18,7% |
  | S1: kawasan 4 meter | +6,4% | +3,2% | +7,1% |
  | S3: faktor k | 0,505 | 0,657 | tidak dipakai |
  | S4: batas | batas latih | batas latih | batas latih |

  S3 menarik turun tebakan gas di dekat batas 999.765 kWh; S4 memotong prediksi ke batas data latih; 77 kawasan uji
  yang identik dengan kawasan latih disalin. *Leaderboard* publik 0,24920 menjadi 0,24913. Reprodusibilitas: *seed*
  dikunci, mode deterministik, sidik sha256 aec745ab…d66e tercetak di sel 35. Pelatihan penuh sekitar 20 jam; tanpa
  pelatihan sekitar 5 jam.
- **Pertanyaan juri:** "Apa saja langkah setelah penggabungan?" · "Bisakah hasilnya direproduksi?"
- **Pedoman jawaban:** Perakitan berjalan dengan urutan tetap: gabung, koreksi, sesuaikan batas, salin duplikat,
  periksa, lalu tulis. S1 berasal dari prediksi *out-of-fold* penuh, k pada S3 dari *holdout set*, dan batas S4 dari
  data latih; semuanya dibekukan di satu berkas konfigurasi (D015).
- **Sumber:** sel 0, 22, 30, 35, 37; `deck/fakta_turunan.md`; `reports/DECISIONS.md` D015

### C9. Adopsi di subzona baru dimodelkan sebagai peluang dari jumlah rumah tangga
- **Visual atau bukti:** Diagram N dan K: kolom komposisi pemanas adalah pecahan jumlah rumah tangga N, dan bilangan
  bulat terkecil yang membuat semua pecahan bulat adalah N; K adalah rumah tangga di luar kategori pemanas yang
  tercatat, yaitu pengguna pemanas listrik ditambah satu kategori lain yang tidak tercatat. Tabel skor validasi lima
  *fold* per subzona yang dibobot ulang ke komposisi subzona baru: `q_xent` 1,0329; `r_reg` 1,0351; rata-rata keduanya
  1,0303 poin (0,143 setelah dibagi σ adopsi).
- **Pertanyaan juri:** "Bagaimana memprediksi adopsi di subzona yang belum pernah terlihat?" · "Mengapa dua LightGBM?"
  · "Mengapa 8,14%?"
- **Pedoman jawaban:** Adopsi seragam dalam satu subzona, sehingga 91,86% kawasan uji cukup disalin dari subzonanya.
  Untuk 8,14% sisanya, `q_xent` menebak peluang rumah tangga K berada di kategori lain yang bukan pemanas listrik
  (*cross-entropy* dengan bobot K), sedangkan `r_reg` menebak porsi kategori itu secara langsung. Adopsi adalah sisa
  porsi dikurangi tebakan tersebut, dan rata-rata kedua model lebih baik daripada masing-masing. Angka 8,14% adalah
  porsi kawasan uji, bukan porsi subzona, dan sesuai harapan pengacakan per baris (8,17%).
- **Sumber:** sel 28, 35, 39, 54; `final/nbrun_final/dicefest_nb/src/adopt_v2.py` (docstring dan `fit_predict`);
  *writeup* B 5.1

### C10. Urutan zona tahan derau acak; zona contoh tetap pada kedua skenario COP
- **Visual atau bukti:** Tiga blok.
  - Uji ketahanan: derau prediksi acak per kawasan uji (200 simulasi) memberi Spearman 0,998, median 8 zona pindah
    kuadran, dan 5 zona teratas tetap. COP 2,5 sampai 3,5 menurut iklim zona memberi Spearman 0,81, 68 zona pindah
    kuadran, dan hanya 1 dari 5 zona teratas tetap. Galat yang berkelompok antartetangga belum diuji. Asumsi seragam
    hanya menskalakan angka (median +71,1% sampai +128,0%), tidak mengubah urutan.
  - Posisi zona contoh pada dua skenario. Setiap sel ditulis "COP 3 seragam → COP per iklim". Peringkat dihitung
    dari 330 zona, untuk kenaikan konsumsi listrik dalam persen dan untuk tambahan konsumsi dalam TWh. Kolom Peran
    memakai kode rekomendasi ("awal R1" berarti titik awal R1). Baris tebal adalah zona yang dipakai sebagai titik
    awal atau contoh di *slide* utama:

  | Zona | Peran | Kuadran (COP 3 → per iklim) | Peringkat kenaikan % | Peringkat tambahan TWh |
  |---|---|---|---|---|
  | REG_273 | awal R1 | Q1 → Q1 | 47 → 32 | 2 → 2 |
  | REG_270 | awal R1, R2 | Q1 → Q1 | 5 → 1 | 4 → 4 |
  | REG_045 | awal R1, R2 | Q1 → Q1 | 9 → 3 | 3 → 3 |
  | REG_263 | awal R2 | Q1 → Q1 | 25 → 51 | 1 → 1 |
  | REG_290 | skrining R2 | Q1 → Q1 | 1 → 47 | 46 → 60 |
  | REG_259, REG_275 | skrining R2 | Q1 → Q1 | 2, 3 → 6, 7 | 25, 53 → 17, 41 |
  | REG_033 | contoh R3 | Q2 → Q2 | kenaikan +84,92% → +78,37% | |

  - Sebanyak 33 dari 47 zona Q2 dan 98 dari 118 zona Q1 tetap di kuadrannya. Indikator potensi *retrofit* relatif
    terhadap konsumsi gas: Q2 6,6%, Q1 5,5%. Kedua butir ini dan REG_249 ada di pedoman jawaban, tidak di layar.
- **Pertanyaan juri:** "Seberapa yakin dengan urutan zona?" · "Mengapa bukan REG_290, zona dengan kenaikan
  tertinggi, atau REG_259 dan REG_275?" · "Mengapa pompa panas dulu di Q2, padahal potensi *retrofit* relatifnya
  lebih tinggi?" · "Apakah 7,39 TWh realistis?"
- **Pedoman jawaban:** Urutan zona hampir tidak terpengaruh derau acak karena setiap zona merata-ratakan banyak
  kawasan, tetapi sangat terpengaruh COP yang berbeda per iklim. Karena itu titik awal dipilih dari zona yang tetap
  sepuluh besar dalam persen dan tambahan absolut pada kedua skenario, yaitu REG_270 dan REG_045, ditambah REG_263
  sebagai tambahan absolut terbesar. Enam zona tetap sepuluh besar dalam persen, tetapi REG_259 dan REG_275 menambah
  konsumsi listrik yang relatif kecil; keduanya, juga REG_290, tetap dinilai sebagai bagian daftar skrining. Urutan
  program di Q2 mengikuti
  tekanan jaringan: kenaikan listriknya di bawah median, sehingga elektrifikasi di sana menambah tekanan lebih
  sedikit daripada di Q1 dan Q3, sedangkan audit energi tetap berjalan di zona pilot. Angka 7,39 TWh adalah indikator dengan bias dua arah:
  definisinya ikut memuat ukuran rumah yang tidak dapat diubah *retrofit* (bila hanya periode bangunan dihitung,
  tersisa 0,513 bagian), tetapi acuannya rata-rata nasional, sehingga acuan praktik yang lebih baik akan membuatnya
  lebih besar. Audit energi yang menentukan angka nyatanya. Asumsi seragam hanya menskalakan angka (median +71,1%
  sampai +128,0%) tanpa mengubah urutan, dan galat yang berkelompok antartetangga belum diuji. Sebanyak 33 dari 47
  zona Q2 dan 98 dari 118 zona Q1 tetap di kuadrannya. REG_249, zona Q2 di tabel sel 50, pindah ke Q1 pada COP per
  iklim (+90,07% menjadi +95,64%, median 94,60%), sehingga tidak dipakai sebagai contoh pilot.
- **Sumber:** `deck/ppt02/bukti/cek_zona_contoh.json` (kuadran dan peringkat zona contoh, 33 dari 47, 98 dari 118);
  `deck/ppt01/bukti/cek_kebijakan.json` (200 simulasi derau, Spearman, 68 zona); sel 49 sampai 51 (REG_249 di tabel
  sel 50); *writeup* 7.3 dan Tabel 4

### C11. Target efisiensi dan target emisi perlu dipisahkan
- **Visual atau bukti:** Dua panel dependensi SHAP ukuran rumah (`img/dep_ukuran_dua.png`): sumbangan gas per
  m² sekitar +65% di kawasan rumah terkecil, sedangkan sumbangan listrik per rumah sekitar +75% di kawasan rumah
  terbesar (pita 5% ekstrem; antarkuartil +34% dan +35%). Tabel pemetaan target ke trilema dari *writeup* bab 1: gas
  total ke emisi, listrik total ke beban jaringan, EUI ke emisi dan biaya, porsi bangunan bersih ke emisi, adopsi ke
  transisi.
- **Pertanyaan juri:** "Apakah satu program bisa sekaligus menurunkan tagihan dan emisi?" · "Bagaimana lima target
  lomba berhubungan dengan trilema?"
- **Pedoman jawaban:** Ini rekomendasi kelima *notebook*. Rumah kecil boros gas per meter persegi, sedangkan rumah
  besar boros listrik per rumah, sehingga program keterjangkauan sebaiknya menyasar kawasan rumah kecil dan bangunan
  lama, sedangkan program pengurangan emisi menyasar zona dengan volume gas terbesar, seperti REG_263 dan REG_273. Pola
  ini berupa sumbangan pada prediksi model penjelas, bukan efek kausal.
- **Sumber:** sel 47, 48, 51 butir 5; *writeup* bab 1 dan 7.4 butir 5; `deck/ppt01/bukti/cek_kebijakan.json`;
  `deck/ppt02/bukti/cek_zona_contoh.json`

### Pembagian penjawab

| Topik pertanyaan | *Slide* cadangan | Penjawab |
|---|---|---|
| kebocoran, validasi, *overfit* | C3, C7 | Orang 2 |
| jumlah model, bobot adaptif, koreksi, reprodusibilitas | C1, C2, C4, C8 | Orang 2 |
| adopsi di subzona baru | C9 | Orang 2 |
| SHAP dan sebab-akibat | C5 | Orang 3 |
| data yang dibutuhkan, pelaksana, validasi lapangan | C6 | Orang 3 |
| urutan zona, zona contoh, potensi *retrofit* | C10 | Orang 3 |
| efisiensi dan emisi, pemetaan trilema | C11 | Orang 3 |
| pertanyaan umum dan penggunaan alat bantu AI | `06_qna_juri_v2.md` bagian C | Orang 1 |

## E. Konflik angka atau definisi dan hal yang perlu dikonfirmasi

| No. | Konflik atau celah | Temuan di sumber | Penanganan di *deck* v3 |
|---|---|---|---|
| 1 | Arahan menyebut "8,14% subzona baru" | 8,14% adalah porsi kawasan uji (baris) yang berada di subzona tanpa baris latih (sel 35, 54), bukan porsi jumlah subzona | ditulis "8,14% kawasan uji di subzona baru" |
| 2 | Arahan menyebut total 43 model, sedangkan *notebook* menulis "42 model dilatih, 41 masuk ke *ensemble*" | 43 = 41 anggota gabungan + 2 LightGBM adopsi; *notebook* melatih 42 di jalur 1 dan 2, dan `mk1_gpm` tidak terpilih (sel 32). Model penjelas SHAP (4 LightGBM) berada di luar 43 karena tidak menghasilkan prediksi submisi | *slide* 6 memakai 43 = 41 + 2; perbedaan dengan 42 disiapkan di C4 |
| 3 | "41 model di-*blend*" dapat terbaca sebagai satu gabungan, dan 41 dapat terbaca sebagai 41 jaringan | ada dua kelompok anggota (28 dan 13) dan tiga set bobot (gas, EUI, listrik); 41 dihitung per keluaran karena MLP dua keluaran `mk2` menyumbang ke kedua jalur | *slide* 6 menyebut "gabungan bobot adaptif per jalur"; C1 dan C4 merinci |
| 4 | *Notebook* dan *writeup* menyebut potensi *retrofit* 7,39 TWh "konservatif", sedangkan *deck* v2 menyebutnya batas atas | keduanya hanya benar sebagian: definisinya ikut memuat ukuran rumah (periode bangunan saja 0,513 bagian), tetapi acuannya rata-rata nasional, sehingga acuan yang lebih ketat membuatnya lebih besar (*writeup* 7.3) | *deck* v3 menyebutnya "indikator potensi, bukan janji penghematan" dan menjelaskan bias dua arah di C10 |
| 5 | *Notebook* sel 51 butir 4 dan *writeup* 7.4 butir 4 menyebut adopsi "hampir hanya" terjadi di kawasan yang kurang tersambung gas; sel 0, sel 48 butir 4, dan *writeup* 7.2 memakai kata "terkonsentrasi" | tingkat adopsi memang 40,24% vs 2,73%, tetapi sebagian besar pengguna pemanas listrik tinggal di kawasan bergas 60% ke atas (47,04% + 35,88% rumah tangga) | *deck* memakai perbandingan tingkat adopsi, bukan klaim "hampir hanya" atau "terkonsentrasi" |
| 6 | Label Q2 di keluaran sel 49, "jaringan masih longgar" | sel 48 menyatakan label itu berasal dari versi lama karena kapasitas jaringan tidak ada di data | Q2 dibaca sebagai "kenaikan listrik relatif lebih rendah" |
| 7 | Sel 51 menyatakan peringkat zona tidak berubah oleh perubahan asumsi | hanya berlaku untuk asumsi seragam (aljabar); COP per iklim memindahkan 68 zona. Uji derau memakai derau acak per kawasan; galat berkelompok belum diuji | *deck* memakai hasil uji ketahanan dengan cakupannya (*slide* 8, C10) |
| 8 | "Peta 330 zona" dapat diartikan peta geografis | data tidak memuat koordinat; yang tersedia peta kuadran | *slide* 8 menyebut "peta kuadran, bukan peta geografis" |
| 9 | Pemetaan target ke aspek trilema berbeda antara *deck* v2 *slide* 2 dan *writeup* bab 1 | *writeup*: porsi bangunan bersih ke emisi dan adopsi ke transisi; *deck* v2: porsi bersih ke biaya dan adopsi ke emisi | *deck* v3 mengikuti *writeup*: R1 biaya dan emisi, R2 keandalan, R3 transisi; tabelnya di C11 |
| 10 | *Writeup* menulis "rata-rata 95%", sel 50 mencetak median 94,85% | rata-rata 94,74% dan median 94,85% sama-sama dibulatkan menjadi 95% | *deck* memakai "median +94,85%" di layar dan "hampir 95 persen" di narasi |
| 11 | Kontras ukuran rumah | teks sel 48 dan *writeup* 7.2 menulis "sekitar 65%" untuk rumah "sekitar 30 m²" vs "300 m²", yaitu ujung garis rata-rata grafik; hitungan pita 5% memberi 66,5% untuk rata-rata ±37 vs ±173 m² | *deck* memakai "+65%" sesuai *notebook* dengan ukuran pita 5% (±37 dan ±173 m²), ditambah kontras antarkuartil +34% |
| 12 | Urutan nomor subzona | sel 54 dan *writeup* 4.3 memakai 99,07% per baris; 73,39% pasangan baris berurutan berada di subzona yang sama sehingga otomatis satu zona. Ukuran yang informatif adalah 96,5% pasangan subzona bernomor berurutan (163.185 subzona) | *deck* memakai 96,5% dan menyebut nomor berdekatan sebagai pendekatan kedekatan lokasi |
| 13 | Sel 39 menyebut koreksi galat wilayah menurunkan *leaderboard* 0,0007 (v12 ke v15) | koreksi tingkat grup sudah ada sejak v9; langkah v12 ke v15 juga memuat model adopsi, MLP dua keluaran, salinan duplikat, dan pencabutan CatBoost; bagian kNN sekitar 0,0004 | C2 memakai angka bagian kNN dan menjelaskan gabungan langkahnya |
| 14 | Zona awal *notebook* dan *deck* v2 tidak tahan skenario COP per iklim | REG_290 turun dari peringkat 1 ke 47 kenaikan relatif; REG_249 pindah dari Q2 ke Q1 (95,64% vs median 94,60%) | R2 dimulai dari REG_263, REG_270, dan REG_045; contoh R3 diganti REG_033 yang tetap Q2; rincian di C10 |
| 15 | `deck/ppt01/fakta_tambahan.md` bagian A menulis label "kelima target" diacak | skrip uji mengacak kolom turunan tiga target konsumsi (gas, listrik, EUI) yang dipakai fitur | *deck* v3 menulis "label target konsumsi"; berkas *deck* v2 tidak diubah |
| 16 | Sel 48 butir 4 memakai porsi sosiodemografi 15% sebagai bukti bahwa adopsi tidak terutama dijelaskan karakteristik warga | 15% diukur terhadap total yang 81%-nya identitas komposisi (*writeup* Tabel 3), sehingga tafsir tersebut tidak didukung | angka 15% tidak dipakai di *slide* utama; catatan di C5 |
| 17 | Nama tahap koreksi | *notebook* memakai "koreksi galat wilayah", arahan memakai "koreksi galat tetangga" | *deck* memakai "koreksi galat tetangga"; C2 menyebut nama di *notebook* |
| 18 | Nama pembicara di kalimat serah terima akhir *slide* 2, 7, dan 11 | nama anggota tim tidak tercantum di repositori; v3 memakai penanda "[nama Orang 1]" sampai "[nama Orang 3]" yang terbaca sebagai nama salah di catatan pembicara | v4 memakai kalimat tanpa nama ("rekan kami ..."); peran tetap tertulis di baris waktu catatan (Orang 1 jangkar, Orang 2 teknis, Orang 3 kebijakan) |
| 19 | [PERLU KONFIRMASI: apakah panduan final mewajibkan semua anggota berbicara, apakah slot 10 menit sudah termasuk waktu berbagi layar di Zoom, dan jam tampil] | panduan final belum terbaca dari server | pembagian tiga pembicara dan *buffer* 30 detik disiapkan untuk kedua kemungkinan; sapaan pembuka dibuat netral tanpa "pagi" atau "siang" |
| 20 | [PERLU KONFIRMASI: biaya *retrofit*, tarif energi, dan intensitas karbon listrik] | tidak ada di data | *deck* tidak mengklaim manfaat ekonomi, penghematan rupiah, atau penurunan emisi bersih |
| 21 | Pemetaan gardu berbeda di *deck* v3: *slide* 2 menulis "gerbang untuk pompa panas di Q1 dan Q2", sedangkan tabel *slide* 8 memberi R2 ke Q1 dan Q3 | sel 51 butir 3: operator jaringan dilibatkan sejak awal pada zona Q1 dan Q3, yaitu zona dengan kenaikan listrik di atas median; butir 2: kesiapan gardu Q2 cukup dikonfirmasi sebelum pilot. Ketiga zona awal R2 berada di Q1 pada kedua skenario COP (`make_visuals_v4.py` memeriksanya) | v4 memakai pemetaan tabel *slide* 8 di *slide* 2, 8, 10, 13, dan C6: R1 di Q1, R2 di Q1 dan Q3, R3 di Q2; cek gardu lokal tetap menjadi langkah pertama pilot R3 |
| 22 | Rujukan berkas bukti di catatan pembicara v3 ditulis tanpa folder (`cek_kebijakan.json`, `bukti/cek_zona_contoh.json`) | dari folder v4, `bukti/` tidak ada; berkasnya di `deck/ppt01/bukti/` dan `deck/ppt02/bukti/` | v4 menulis jalur lengkap dari akar repositori |
| 23 | *Slide* 23 v3 (C10) memuat rujukan "contoh *deck* v2" yang tidak pernah dilihat juri, kuadran REG_263 kosong, REG_273 tidak ditebalkan padahal titik awal R1, dan sel "peringkat 1 dan 4" tidak menyebut ukurannya | `cek_zona_contoh.json` dan tabel sel 50 | v4: REG_249 disebut sebagai Q2 di tabel sel 50, setiap sel memuat kuadran dan ukuran peringkat ("relatif", "absolut"), dan empat titik awal serta contoh pilot ditebalkan |

## F. Daftar verifikasi akhir

Butir yang dapat dihitung diperiksa oleh `cek_struktur.py` (hasil lengkap di blok di bawah). Butir lain diperiksa
dengan membaca ulang dokumen dan diaudit oleh pemeriksa berkonteks bersih.

<!-- hasil-cek:mulai -->
- [x] **Total durasi bicara tepat 9 menit 30 detik.** jumlah durasi 13 *slide* 570 detik; rentang waktu bersambung 00:00 sampai 09:30; total per bagian 60, 150, 90, 210, dan 60 detik.
- [x] ***Buffer* 30 detik berada di luar narasi utama.** narasi berakhir di 09:30; 09:30 sampai 10:00 hanya untuk serah terima dan jeda (bagian B2).
- [x] **Pembukaan mengungkapkan Trilema Energi dan tiga rekomendasi dalam 60 detik.** *slide* 1 dan 2 berdurasi 60 detik dengan 132 kata lisan (132 kata per menit); trilema di *slide* 1, tiga rekomendasi di *slide* 2; tanpa istilah model.
- [x] **Pembahasan mengikuti urutan rekayasa fitur, arsitektur model, kebijakan, lalu keterbatasan.** urutan bagian rubrik per *slide*: I, I, II, II, II, III, III, IV, IV, IV, IV, V, V.
- [x] **Total 43 model dibedakan jelas menjadi 41 model yang di-*blend* dan 2 LightGBM terpisah.** *slide* 6 memuat 43, 41, 28, 13, "2 LightGBM", dan "terpisah"; tidak ada klaim 42 model atau satu gabungan 43 model di *slide* utama.
- [x] **Peta 330 zona, prioritas Q1, kapasitas gardu, dan pilot elektrifikasi bergas dibahas sesuai bukti.** *slide* 8 sampai 11; *slide* 9 sampai 11 memakai rantai TEMUAN, INTERPRETASI, REKOMENDASI, LANGKAH.
- [x] **Frasa "temuan model mendukung rekomendasi ini" digunakan pada bagian kebijakan publik.** 3 kali, di narasi *slide* 9, 10, 11.
- [x] **Tidak ada klaim kausalitas mutlak berdasarkan SHAP.** 0 kata kausal (menyebabkan, mengakibatkan, membuktikan, terbukti, dan sejenisnya) di judul, isi, dan narasi *slide* utama; SHAP selalu disebut sumbangan pada prediksi atau asosiasi.
- [x] **Tidak ada angka atau hasil empiris yang dikarang.** 530 angka di bagian A, C, D, dan E dicocokkan dengan enam lembar fakta: 0 tidak ditemukan; 0 angka baru di menit terakhir.
- [x] **Detail teknis ekstrem hanya muncul di *slide* cadangan Q&A.** 0 istilah rinci di *slide* utama; di *slide* cadangan: `A2xM2xp2`, `NNLS`, `PLR`, `Poisson`, `TabM`, `TreeSHAP`, `early stopping`, `n/(n`, `q_xent`, `r_reg`, `λ`.

Pemeriksaan tambahan:

- tidak ada em-dash (U+2014): ya
- 13 *slide* utama dan 11 *slide* cadangan; setiap *slide* memuat keempat komponen
- kecepatan narasi setiap *slide* di rentang 120 sampai 140 kata lisan per menit: ya
- sebutan "batas atas" untuk potensi *retrofit* di *slide* utama: tidak ada
- pemetaan gardu R2 di Q1 dan Q3 sama di *slide* 2, 8, 10, dan 13: ya
- baris PELAKSANA di *slide* 9, 10, dan 11: ya
- penanda nama pembicara di narasi: tidak ada
- nomor sel *notebook* untuk baris kaki di setiap *slide* utama: ya
<!-- hasil-cek:selesai -->

## G. Perubahan dari *deck* v2 dan dari versi pertama v3

### G1. Dari *deck* v2 (ppt01)

| Aspek | *Deck* v2 | *Deck* v3 | Alasan |
|---|---|---|---|
| Alur | masalah, data, fitur, model, validasi, SHAP, peta, rekomendasi, batas | ringkasan eksekutif, fitur, arsitektur, kebijakan, batas | *top-down*: keputusan dipahami sebelum metode |
| Menit pertama | kesimpulan dan masalah dalam 65 detik | trilema, tesis, dan tiga rekomendasi dalam 60 detik tanpa istilah model | arahan bagian I |
| Struktur data (identitas target) | *slide* tersendiri | dilebur ke narasi *slide* 6 sebagai alasan tiga jalur | menghemat waktu tanpa kehilangan argumen |
| SHAP | *slide* pendorong konsumsi tersendiri | model penjelas didefinisikan di *slide* 3, SHAP dibaca langsung di *slide* rekomendasi 1 dan 3 | setiap grafik teknis langsung bermuara pada tindakan |
| Jumlah model | 28, 13, dan 2 LightGBM tanpa total | 43 = 41 di-*blend* + 2 terpisah, dengan kurung pemisah di diagram | arahan bagian III |
| Rekomendasi | kartu siapa, apa, ukuran, dasar bukti | rantai temuan, interpretasi, rekomendasi, langkah, dengan frasa "temuan model mendukung rekomendasi ini" | arahan bagian IV |
| Zona contoh | REG_290 untuk gardu, REG_249 untuk pilot | REG_263, REG_270, REG_045 untuk gardu, REG_033 untuk pilot | tetap pada kedua skenario COP |
| Keterbatasan | *slide* 13 dengan angka | *slide* 12 tanpa angka; angka uji ketahanan sudah muncul di *slide* 8 | menit terakhir tanpa klaim baru |
| Durasi dan jumlah *slide* | 9 menit 20 detik, 14 + 8 *slide* | tepat 9 menit 30 detik, 13 + 11 *slide* | arahan bagian 2 |

### G2. Dari versi pertama v3, setelah audit independen

| Temuan audit | Perubahan |
|---|---|
| Uji baris latih tertulis "0 baris berubah" tanpa konteks (berat) | *slide* 5 dan C3 menyebut fitur bagian yang diacak identik, sedangkan 309.097 baris bagian lain berubah sesuai rancangan |
| Model penjelas dan SHAP tidak didefinisikan di narasi utama (berat) | definisi model penjelas di *slide* 3, definisi SHAP di *slide* 9 |
| "Batas atas" tidak terbukti dan bertentangan dengan *writeup* | diganti "indikator potensi" dengan bias dua arah (C10, E4) |
| REG_290 dan REG_249 tidak tahan skenario COP per iklim | zona awal dan contoh diganti; bukti di `bukti/cek_zona_contoh.json` |
| 99,07% sebagian besar mekanis | diganti 96,5% pasangan subzona; "pendekatan kedekatan lokasi" |
| Klaim bebas kebocoran melampaui cakupan uji | judul C3, piramida B1, dan label cakupan *slide* 5 dibatasi pada dua blok fitur dan satu *fold* |
| Sosiodemografi 15% dipakai sebagai bukti | dihapus dari *slide* 11; catatan di C5 |
| Ambang median, makna kolom, dan nilai prediksi belum diakui | ditambahkan di *slide* 12, A5, dan C6 |
| Sudut trilema hilang setelah *slide* 1; rekomendasi 5 tanpa cadangan | label trilema di *slide* 2 dan 13; C11 baru |
| Panah 1 → 2 → 3 menyiratkan satu urutan waktu | gardu digambar sebagai gerbang; judul *slide* 13 "diurutkan per kuadran" |
| Mekanisme selubung tidak diucapkan; urutan narasi R1 terbalik | narasi *slide* 9 mengikuti temuan, interpretasi, rekomendasi, langkah |
| Q2 berpotensi *retrofit* relatif lebih tinggi | audit energi masuk langkah pilot; jawaban di C10 |
| Uji derau hanya acak per kawasan | ditulis "tahan derau acak per kawasan"; galat berkelompok disebut belum diuji |
| Penjelasan `q_xent` dan `r_reg` terbalik | C9 diperbaiki: model menebak kategori lain yang bukan pemanas listrik |
| Atribusi 0,0007 di C2 | C2 dan E13 memakai bagian kNN sekitar 0,0004 |
| Temuan ringan lain | label matriks *slide* 10, label target konsumsi, keterangan skala galat, "di dua jalur ini", sapaan netral, *notebook* 56 sel dan 17 modul di *slide* 7, cetak miring istilah asing |

### G3. Saat PPTX v3 dibangun (28 Oktober 2026)

| Temuan | Perubahan |
|---|---|
| Kalimat "yang tetap tinggi di kedua skenario adalah REG_270 dan REG_045" tidak eksklusif: enam zona tetap sepuluh besar kenaikan relatif pada kedua skenario (REG_045, REG_259, REG_260, REG_270, REG_275, REG_330) | kriterianya ditulis tepat, yaitu sepuluh besar dalam persen dan tambahan absolut pada kedua skenario, yang hanya dipenuhi REG_270 dan REG_045 (`img/data_grafik.json`); A3, *slide* 10, dan C10 diperbarui, dan C10 mendapat baris REG_259 dan REG_275 |
| Grafik *deck* v2 memuat REG_290 dan label kuadran versi lama | peta kuadran dan plot SHAP digambar ulang oleh `make_visuals_v3.py`; grafik batang menjadi grafik asli PowerPoint dari `img/data_grafik.json` |
| Label angka bawaan grafik mengikuti pengaturan bahasa komputer presentasi, sehingga koma desimal dapat berubah menjadi titik | label grafik ditulis sebagai kotak teks di posisi batang yang dihitung dari area plot tetap |

### G4. *Deck* v4 (28 Oktober 2026, atas permintaan pengguna)

Pengguna meminta perbaikan pada v3 dengan mengambil elemen dari v2. Permintaan dikelompokkan per kriteria penilaian.
Setiap angka yang baru masuk *deck* diperiksa oleh `make_visuals_v4.py` atau tercantum di lembar fakta.

| Kriteria | Permintaan | Perubahan di v4 | *Slide* |
|---|---|---|---|
| 2 (arsitektur) | judul *slide* 6 dari v2 yang menonjolkan dekomposisi bersih | "Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik" | 6 |
| 2 | label "Jalur 3" yang salah | v3 menulis "Jalur 3 · Adopsi", padahal jalur ini juga memuat porsi bersih; kini "Jalur 3 · Porsi bersih dan adopsi: tanpa model besar" | 6 |
| 2 | kartu sidik sha256 di *slide* 7 diganti grafik *leaderboard* per langkah dari v2 | grafik garis asli PowerPoint dari keluaran sel 37 dengan titik skor privat; sidik sha256 pindah ke baris kaki | 7 |
| 2 | nomor sel *notebook* di baris kaki *slide* utama | medan "Sel notebook" di setiap *slide* utama, dicetak 12 pt di kanan bawah | 1 sampai 13 |
| 1 (pra-pemrosesan) | banner bawah *slide* 6 diganti tiga fakta struktur data dari v2 | pita tiga fakta (gas = EUI × luas lantai, porsi bersih = rumus, adopsi seragam per subzona) dengan angka buktinya dari sel 25; "43 = 41 + 2" pindah ke sudut jalur 3 | 6 |
| 1 | porsi rasio fisik 25–30% dan penanda fitur turunan | angka 25–30% (*writeup* B Tabel 3, dihitung ulang dari nilai SHAP) dan batang fitur turunan berwarna sendiri dengan legenda | 3 |
| 4 (kerapihan) | rata-rata kata per *slide* paling banyak 100 | teks dipangkas, terutama *slide* cadangan; baris "Menjawab" pindah ke catatan; hasil hitung di `cek_pptx.py` | semua |
| 4 | label di bawah 12 pt naik ke minimal 14 pt | semua teks di layar minimal 14 pt, kecuali baris kaki 12 pt; teks di dalam gambar juga 14 pt pada ukuran tampilnya | semua |
| 4 | kesalahan di *slide* 13 | langkah berikutnya kini sejajar dengan kartu rekomendasinya (v3: langkah gardu berada di bawah kartu *retrofit*), kartu gardu menyebut Q1 dan Q3 | 13 |
| 4 | kesalahan di *slide* 23 | lihat bagian E butir 23: rujukan "*deck* v2" dihapus, kuadran REG_263 diisi, ukuran peringkat ditulis, titik awal ditebalkan, judul dirapikan | C10 |
| 4 | nama yang salah di catatan pembicara | penanda "[nama Orang N]" diganti kalimat serah terima tanpa nama; jalur berkas bukti ditulis lengkap (bagian E butir 18 dan 22) | 2, 7, 11, catatan |
| 3 (kebijakan) | dasar SHAP untuk rekomendasi 2 dengan grafik listrik per rumah +75% dari v2 | panel SHAP listrik per rumah digambar ulang (sumbu v2 salah cetak 0,25 menjadi "0,2") dan dipakai di kotak TEMUAN serta narasi | 10 |
| 3 | grafik SHAP sebagai visual utama *slide* 11 | grafik SHAP adopsi versi besar menggantikan diagram batang; angka tingkat adopsi tetap di TEMUAN | 11 |
| 3 | satu baris "Pelaksana:" per rekomendasi | baris PELAKSANA di bawah rantai *slide* 9, 10, dan 11, sama dengan tabel C6 | 9, 10, 11 |
| 3 | pemetaan gardu *slide* 2 sama dengan tabel *slide* 8 | R2 di Q1 dan Q3 pada *slide* 2, 8, 10, 13, dan C6 (bagian E butir 21) | 2, 8, 10, 13 |

### G5. Setelah tinjauan independen v4 (28 Oktober 2026)

Pemeriksa berkonteks bersih menilai kesepuluh permintaan sudah dikerjakan, semua angka di layar cocok dengan lembar
fakta, dan pemetaan kuadran konsisten. Pemeriksa juga melaporkan 18 temuan. Temuan yang terverifikasi ke sumber
diperbaiki sebagai berikut:

| Temuan | Perbaikan |
|---|---|
| Baris kaki memakai indeks sel dari 0 (ikut menghitung markdown), padahal juri melihat nomor eksekusi In [n] dan judul bagian; misalnya sel 35 adalah In [27] | baris kaki kini memakai nomor bagian dan In [n]; tabel padanan indeks dan In [n] ada di B4; *slide* 5 menandai uji label acak sebagai skrip terpisah |
| Kolom galat *slide* 6 terbaca sebagai galat akhir | judul kolom "Galat jujur sebelum koreksi (RMSE/σ; menebak rata-rata ≈ 1)", dan narasi menyebutnya |
| C9: N disebut rumah tangga satu kawasan | "N rumah tangga di satu subzona" (sel 28) |
| C8: perbaikan *leaderboard* 0,24920 menjadi 0,24913 dikaitkan dengan salinan duplikat | dikaitkan dengan S1, S3, dan S4 (sel 37, langkah "penyesuaian meter sedikit + batas gas") |
| *Slide* 9: judul grafik tidak menyebut satuan batang | "(GWh; % gas zona)" |
| *Slide* 6: "galat maks" tanpa "relatif" | "galat relatif maks" |
| *Slide* 7: langkah v12 ke v15 berlabel "koreksi tetangga", padahal memuat beberapa perubahan | label "koreksi dan lainnya"; pedoman C2 merinci isinya |
| C7: "tidak *overfit*" mutlak | "konsisten dengan tidak *overfit*"; *leaderboard* hanya konfirmasi arah |
| Narasi *slide* 7 dan 10 lebih kuat dari bukti | "di perangkat yang sama"; REG_263 disebut karena tambahan absolutnya |
| C10: Q2 disebut paling sedikit menambah tekanan, padahal Q4 lebih rendah | "lebih sedikit daripada di Q1 dan Q3" |
| *Slide* 9: "+65%" tanpa "sekitar" | "sekitar +65%" |
| *Slide* 2: kotak Q3 biru tua hilang di kartu biru tua | kotak sasaran berwarna sama dengan kartu ditulis putih |
| Kata terpotong di tanda hubung atau tertinggal sendiri (*slide* 5, 6, 9, C3) | pemenggalan baris diatur; label "median 330 zona" utuh |
| C5: porsi kelompok fitur tidak berjumlah 100% | catatan "sisa porsi tiap baris: kelompok lokasi dan agregat kawasan" |
| *Slide* 12: "SHAP asosiatif" berlabel BELUM DIKETAHUI | "Efek sebab-akibat dan penghematan nyata: belum terukur" |
| *Slide* 11 dan C6: langkah pilot berbeda | audit energi disebut di narasi *slide* 11 dan di C6 |
| Rujukan bahan tanya jawab lama | bagian D merujuk `deck/ppt00/03_qna_juri.md` dan mencatat bahwa nomor *slide* di bank tanya jawab v2 berbeda |

Tabel C10 juga disusun ulang menjadi satu kolom per ukuran dengan format "COP 3 → COP per iklim". Peringkat kenaikan
relatif REG_273 (47 menjadi 32) dicatat di `fakta_ppt03.md`.
