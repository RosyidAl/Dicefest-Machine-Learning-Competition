# Struktur Deck Final v2 (10 menit): Tim BISMILLAH, DICE FEST 2026

Dokumen ini adalah sumber kebenaran isi deck final. Bagian C dan D dibangkitkan otomatis oleh `build_deck_v2.py` dari
data yang sama dengan `BISMILLAH_Final_v2.pptx`, sehingga isi dokumen dan isi slide tidak bisa berbeda. Bagian A, B,
E, dan F ditulis tangan (`brief_kepala.md`, `brief_ekor.md`). Berkas ini menggantikan `../04_struktur_deck.md`.

**Sumber angka:** keluaran notebook `final/deliverables/DICEFEST2026_BISMILLAH.ipynb` (`../fakta_notebook.txt`),
`../fakta_turunan.md`, dan `fakta_tambahan.md`. Nomor sel dihitung dari 0. Setiap slide menyebut nomor selnya di label
bukti.

## A. Perubahan dari deck 04 dan alasannya

| Celah di deck 04 | Perubahan di v2 | Slide |
|---|---|---|
| 1. Naskah 01 dan 02 terlalu panjang dan bersaing dengan narasi 04 | Narasi tiap slide kini satu-satunya naskah lisan; naskah 01 dan 02 ditandai arsip di `README.md` | semua |
| 2a. Pembuka langsung memakai skor dan jargon | Slide 1 berisi kesimpulan (urutan program untuk 330 zona). Slide 2 menjelaskan mengapa masalah layak diselesaikan, dengan risiko kenaikan listrik +94,85% | 1, 2 |
| 2b. Slide arsitektur memuat lima pesan dalam 80 detik | Slide 6 kini satu pesan: tiga jalur untuk lima target, dengan galat jujur per jalur. Rincian bobot adaptif, koreksi, dan penyesuaian akhir pindah ke cadangan C1 sampai C3 | 6, C1, C2, C3 |
| 3. Rekomendasi belum menyebut pelaksana, ukuran keberhasilan, dan tingkat keyakinan | Tiap rekomendasi punya slide sendiri dengan kartu Siapa, Apa, Ukuran, dan Dasar bukti. Tabel peta menghubungkan tiap kuadran ke rekomendasinya. Uji ketahanan menunjukkan urutan zona tahan galat prediksi tetapi peka terhadap COP per iklim, sehingga daftar zona disebut kandidat | 9 sampai 12, C4 |
| 4. Keterbatasan dan penutup menumpang di slide rekomendasi 3 | Slide 13 khusus keterbatasan (tiga hal yang belum diketahui beserta penanganannya), slide 14 khusus penutup (tiga alasan dan tiga langkah berikutnya) | 13, 14 |
| 5. Klaim bebas kebocoran hanya didukung selisih LB 0,00011 | Uji label acak pada modul notebook, untuk baris validasi (42.931) dan baris latih (77.275): 0 baris fitur berubah; kontrol bocor berubah di 42.931 baris | 5, C7 |
| 6. LB publik dipakai sebagai bukti utama | Slide 7 memakai skor privat 0,24844 (peringkat 8) dari data yang tidak pernah terlihat, ditambah AUC adversarial 0,4997 dan 8,14% vs 8,17%; *leaderboard* publik hanya disebut sebagai konfirmasi arah | 7 |
| 7. Lima bagian RISIKO di tanya jawab belum ditutup | P4 ditutup dengan uji label acak; P2 dihitung ulang; P1 didukung bukti pengembangan (D002); P3 dan P5 diberi rujukan audit. Lihat `06_qna_juri_v2.md` | C1, C7 |
| 8. Belum ada deck yang bisa dipresentasikan | `BISMILLAH_Final_v2.pptx` (dapat disunting, dengan catatan pembicara) dan pratinjau PDF | semua |

**Sistem bukti di layar.** Setiap angka penting diberi label kecil yang sekaligus menyebut sumbernya:
- **FAKTA** (biru tua): keluaran notebook atau data;
- **HASIL** (hijau kebiruan): hasil validasi atau model penjelas;
- **ASUMSI** (jingga): asumsi skenario;
- **INTERPRETASI** (ungu): bacaan atau usulan tim.

Label ini menjawab arahan A6 dan C5 (membedakan fakta, asumsi, dan interpretasi) tanpa menambah paragraf.

**Koreksi setelah audit independen (27 Oktober).** Pemeriksa berkonteks bersih menemukan 25 masalah. Yang terpenting
sudah diperbaiki, dan angka barunya ada di `fakta_tambahan.md` bagian F sampai H:
- **Potensi *retrofit*:** 7,39 TWh kini disebut batas atas. Definisinya (sel 48) ikut menghitung ukuran rumah; bila
  hanya periode bangunan dihitung, tersisa sekitar separuh.
- **Rekomendasi 2:** dasarnya kini rumus kenaikan (gas dibanding listrik saat ini) dan tambahan absolut (REG_263),
  bukan "rumah besar".
- **Rekomendasi 3:** kini pilot pompa panas di zona Q2 (REG_249), dengan tingkat adopsi 40% vs 2,73%. Klaim keliru
  "adopsi hampir hanya di kawasan tanpa gas" sudah dihapus.
- **Urutan zona:** klaim "tetap" diganti hasil uji ketahanan. Urutan tahan galat prediksi (Spearman 0,998) tetapi peka
  terhadap COP per iklim (0,81).
- **Slide 7:** *leaderboard* publik hanya sebagai konfirmasi arah (penolakan D011 disebut); privat "tidak memburuk".
- **Slide 8:** ukuran rumah ±37 vs ±173 m² untuk pita 5% ekstrem; kontras kuartil +34% dan +35%.
- **Lain-lain:**
  - judul slide 1 dan 2;
  - label galat jujur "sebelum koreksi" di slide 6;
  - baris notebook di slide 6;
  - "perangkat yang sama" untuk klaim identik byte;
  - bobot 224 + 224 + 104 di C1;
  - asal S1, S3, dan S4 di C3;
  - urutan angka D002 di C7;
  - kata "privat".

**Desain.**
- **Rasio dan huruf:** 16:9, Calibri. Pratinjau memakai Carlito, yang ukuran hurufnya identik.
- **Ukuran teks:** judul 28 pt, isi 14 sampai 18 pt, label dan istilah 11 sampai 12 pt.
- **Warna:**
  - biru tua `#16324F` untuk struktur;
  - hijau kebiruan `#1F7A8C` untuk listrik dan hasil validasi;
  - jingga `#D97B29` hanya untuk angka terpenting dan gas.
- **Slide gelap:** hanya pembuka dan penutup.
- **Dekorasi:** tidak ada garis hias atau ikon dekoratif.

## B. Pembagian waktu dan rubrik

Kecepatan acuan 130 kata per menit. Total rencana 9 menit 20 detik, menyisakan 40 detik dari batas 10 menit untuk
serah terima antarpembicara dan jeda.

| Bagian rubrik (bobot) | Slide | Waktu | Porsi dari isi (tanpa pembuka dan penutup) |
|---|---|---|---|
| Kebijakan dari interpretabilitas (35) | 8, 9, 10, 11, 12 | 3 menit 40 detik | 47% |
| Rekayasa fitur dan pra-pemrosesan (25) | 3, 4, 5, separuh 7 | 2 menit 17 detik | 30% |
| Arsitektur model (15) | 6, separuh 7 | 1 menit 12 detik | 16% |
| Keterbatasan (bagian dari kebijakan) | 13 | 35 detik | 7% |
| Pembuka dan penutup | 1, 2, 14 | 1 menit 35 detik | |

Slide 7 dihitung separuh untuk fitur, karena membuktikan validitas pra-pemrosesan, dan separuh untuk arsitektur, karena
memperlihatkan sumbangan tiap komponen. Bobot slide dan tanya jawab (25) dinilai dari keseluruhan deck dan sesi tanya
jawab. Kesiapan tanya jawab ada di `06_qna_juri_v2.md` dan cadangan C1 sampai C8.

## C. Isi per slide

### Slide 1. Pesan utama (30 detik)
- **Bagian rubrik:** Pembuka
- **Judul:** Dari energi 613 ribu kawasan ke urutan program untuk 330 zona
- **Isi di layar:** Trilema transisi energi perkotaan: emisi, keandalan jaringan, keterjangkauan | *Retrofit* dulu di 118 zona Q1 (47% konsumsi gas) | Gardu dinilai sebelum subsidi pompa panas | Pilot pompa panas di kawasan bergas, mulai zona Q2 | skor *leaderboard* privat, peringkat 8 | fitur bertarget, baris validasi dan latih | hasil sama setiap diulang di perangkat sama
- **Visual:** Slide gelap: judul, tiga kartu rekomendasi bernomor, tiga angka kredibilitas. Angka besar, sel tabel, dan label diagram: 1 | 2 | 3 | 0,24844 | Lulus uji label acak | Identik byte
- **Istilah:** *Retrofit*, pompa panas (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 0, 25, 49, 50; Kaggle LB privat; bukti/uji_kebocoran_*.json
- **Narasi pembicara (±30 detik):** Selamat pagi, Bapak dan Ibu juri. Kami tim BISMILLAH, dan kami mulai dari kesimpulan. Dari lima besaran energi untuk 613 ribu kawasan, 184 ribu di antaranya kami prediksi, kami menyusun urutan program untuk 330 zona: retrofit dulu di zona yang boros gas, kapasitas gardu dinilai sebelum subsidi pompa panas, dan pilot pompa panas di kawasan bergas. Model di baliknya meraih peringkat 8 di leaderboard privat dan lulus uji kebocoran.

### Slide 2. Masalah (35 detik)
- **Bagian rubrik:** Pembuka
- **Judul:** Elektrifikasi penuh hampir menggandakan konsumsi listrik zona, jadi urutan program menentukan kesiapan jaringan
- **Isi di layar:** Tiga tuntutan transisi, diukur oleh lima target lomba | Emisi turun | gas total; adopsi pemanas listrik | Jaringan listrik tetap andal | listrik total | Biaya tetap terjangkau | intensitas gas per m² (EUI); porsi bangunan bersih | median kenaikan konsumsi listrik tahunan zona bila semua pemanas gas diganti pompa panas | konsumsi gas 330 zona (sel 50) | **Pertanyaan kebijakan:** zona mana di-*retrofit* dulu, mana siap dielektrifikasi, dan mana yang jaringannya disiapkan lebih dulu?
- **Visual:** Tiga baris tuntutan dan target di kiri; angka besar +94,85% dengan label asumsi di kanan; kotak pertanyaan kebijakan di bawah. Angka besar, sel tabel, dan label diagram: +94,85% | 141,35 TWh
- **Istilah:** COP, median (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 49, 50 (In [34], In [35])
- **Narasi pembicara (±35 detik):** Mengapa masalah ini layak diselesaikan? Kota yang beralih dari pemanas gas ke pemanas listrik harus memenuhi tiga tuntutan sekaligus: emisi turun, jaringan listrik tetap andal, dan biaya tetap terjangkau. Kelima target lomba mengukur ketiganya. Hasil kami memperlihatkan risikonya. Bila seluruh pemanas gas diganti pompa panas, konsumsi listrik tahunan zona naik dengan median hampir 95 persen. Jadi pertanyaannya bukan apakah kota beralih, tetapi zona mana yang didahulukan dan dengan program apa.

### Slide 3. Struktur data (35 detik)
- **Bagian rubrik:** Rekayasa fitur dan pra-pemrosesan (25)
- **Judul:** Struktur data menjawab dua target sebelum model dilatih
- **Isi di layar:** Gas = EUI × luas lantai | luas lantai = bld_dim_06 | galat relatif maksimum | Satu keluarga model untuk gas dan EUI | Porsi bersih = rumus | 100 × bld_dim_02 / bld_dim_01 | baris latih tepat sama | Dihitung, tidak ditebak | Adopsi seragam per subzona | di 100% subzona data latih | kawasan uji di subzona yang dikenal | Disalin; model hanya untuk 8,14% sisanya
- **Visual:** Tiga kartu temuan, masing-masing dengan angka bukti dan implikasi desain. Angka besar, sel tabel, dan label diagram: 6,7 × 10−16 | 100% | 91,86%
- **Istilah:** EUI, subzona (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 25 (In [22])
- **Narasi pembicara (±35 detik):** Jawaban pertama datang dari struktur data, sebelum model dilatih. Pemeriksaan di sel 25 menemukan tiga hal. Gas selalu sama dengan EUI dikali luas lantai, dengan galat setara presisi mesin, jadi satu keluarga model cukup untuk dua kolom. Porsi bangunan bersih tepat sama dengan satu rumus di seluruh baris latih, jadi dihitung, bukan ditebak. Adopsi pemanas listrik seragam dalam satu subzona, dan 91,86 persen kawasan uji berada di subzona yang sudah dikenal. Model besar hanya dibutuhkan untuk gas, EUI, dan listrik.

### Slide 4. Fitur ukuran rumah (35 detik)
- **Bagian rubrik:** Rekayasa fitur dan pra-pemrosesan (25)
- **Judul:** Luas lantai per meter, proksi ukuran rumah, paling berpengaruh di model penjelas
- **Isi di layar:** area_per_meter = | bld_dim_06 / bld_dim_12 | luas lantai dibagi jumlah meter, kira-kira luas rata-rata satu rumah | dihitung per baris dari dua kolom input | tanpa target dan tanpa statistik lintas baris | rumus sama untuk data latih dan data uji | Tidak dapat membocorkan target | 2,8× dan 3,0× fitur peringkat kedua
- **Visual:** Kartu rumus di kiri; dua diagram batang lima fitur teratas (img/s04_shap_top.png) di kanan. Angka besar, sel tabel, dan label diagram: RUMUS
- **Istilah:** SHAP, model penjelas (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 28, sel 45 (In [32])
- **Narasi pembicara (±35 detik):** Fitur fisik terkuat kami sangat sederhana: luas lantai dibagi jumlah meteran, kira-kira luas rata-rata satu rumah. Gas total sebuah kawasan terutama ditentukan oleh banyaknya rumah, sedangkan ukuran rumah menentukan cara energi dipakai. Pada model penjelas, fitur ini peringkat pertama untuk intensitas gas dan untuk listrik per rumah, hampir tiga kali fitur kedua. Fitur ini tidak mungkin membocorkan jawaban, karena hanya membagi dua kolom input yang juga tersedia di data uji, baris demi baris.

### Slide 5. Fitur kawasan termirip (45 detik)
- **Bagian rubrik:** Rekayasa fitur dan pra-pemrosesan (25)
- **Judul:** Kawasan termirip memakai target tetangga tanpa pernah melihat target baris yang dinilai
- **Isi di layar:** 1 Sepuluh *fold*: satu dinilai, sembilan untuk mencari tetangga | 2 Baris latih: bagian latih dibagi 5, tetangga dari 4 bagian lain | 3 Data uji: tetangga dari bagian latih model yang sama | Jendela ±30 sampai ±300 nomor subzona: 99,07% baris berurutan berada di zona yang sama (sel 54) | Bukti: uji label acak | setelah label 42.931 baris validasi diacak, dan label 77.275 baris latih satu bagian dalam; fitur uji juga identik | versi yang sengaja dibuat bocor: uji ini mampu mendeteksi kebocoran | Galat EUI model tunggal turun 0,0022 (sel 28)
- **Visual:** Skema *cross-fit* dua lapis (bentuk asli, dapat disunting) di kiri; panel hasil uji label acak di kanan. Angka besar, sel tabel, dan label diagram: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | dihitung | bagian 2 | bagian 3 | bagian 4 | bagian 5 | 0 baris berubah | 42.931 baris berubah pada kontrol
- **Istilah:** *out-of-fold*, *fold*, kNN (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 28, sel 54 (In [36]), bukti/uji_kebocoran_*.json
- **Narasi pembicara (±45 detik):** Fitur kedua bertanya: berapa konsumsi kawasan lain yang stok bangunannya paling mirip? Pencarian dibatasi pada jendela nomor subzona, karena nomor subzona ternyata berurutan secara geografis: 99,07 persen baris berurutan berada di zona yang sama. Fitur ini memakai target tetangga, jadi kami menghitungnya out-of-fold dua lapis, seperti skema di kiri. Lalu kami mengujinya: label baris validasi kami acak, lalu label satu bagian dalam baris latih. Fitur yang tidak boleh membaca label itu tetap identik sampai bit terakhir, sedangkan versi yang sengaja dibuat bocor langsung berubah. Jadi uji ini mampu menangkap kebocoran, dan fitur kami lolos.

### Slide 6. Arsitektur tiga jalur (50 detik)
- **Bagian rubrik:** Arsitektur model (15)
- **Judul:** Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik
- **Isi di layar:** Data latih + uji | 201 kolom | Jalur 1 · Gas dan EUI | 28 model: XGBoost, LightGBM, MLP, NCA | Jalur 2 · Listrik | 13 model, pendorong berbeda dari gas | Jalur 3 · Tanpa model besar | porsi bersih: rumus; adopsi: salin subzona (91,86%) + 2 LightGBM (8,14%) | Gas | EUI | Listrik | Porsi bersih | Adopsi
- **Visual:** Diagram alur bentuk asli: data, tiga jalur, tiga tahap setelah penggabungan, kolom galat jujur. Angka besar, sel tabel, dan label diagram: Gabungan bobot adaptif | Koreksi galat wilayah | Penyesuaian akhir | GALAT JUJUR SEBELUM KOREKSI (RMSE/σ) | 0,2449 | 0,6502 | 0,3159 | 0 | ≈ 0,04
- **Istilah:** galat jujur, RMSE/σ (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 23, 31, 35, 39
- **Narasi pembicara (±50 detik):** Karena metrik menilai setiap kolom terpisah, setiap target boleh memakai cara terbaiknya. Jalur pertama untuk gas dan EUI: 28 model, karena keduanya satu besaran. Jalur kedua untuk listrik: 13 model, karena pendorongnya berbeda. Jalur ketiga hampir tanpa model: porsi bangunan bersih dihitung dengan rumus, adopsi disalin dari subzonanya, dan dua LightGBM hanya untuk 8 persen kawasan di subzona baru. Model di dua jalur pertama beragam, lalu digabung, dikoreksi, dan disesuaikan dengan batas data. Angka di kanan adalah galat jujur gabungan sebelum koreksi; koreksi dan penyesuaian menurunkan skor lebih jauh. Notebook-nya 56 sel dengan satu sakelar, dan hasilnya identik byte setiap diulang di perangkat yang sama.

### Slide 7. Teruji di data tersembunyi (45 detik)
- **Bagian rubrik:** Arsitektur model (15) dan validitas pra-pemrosesan
- **Judul:** Skor privat dari data yang tidak pernah terlihat sejalan dengan skor publik, tanda tidak *overfit*
- **Isi di layar:** AUC validasi adversarial: data latih dan uji tidak dapat dibedakan, jadi validasi acak sepuluh *fold* meniru data uji | kawasan uji di subzona baru: teramati vs harapan pengacakan per baris | privat vs publik: tidak memburuk. Komponen diterima di *holdout set*; *leaderboard* hanya mengonfirmasi arah
- **Visual:** Grafik garis skor publik per langkah dan titik skor privat (img/s07_lb.png); tiga bukti di kanan. Angka besar, sel tabel, dan label diagram: 0,4997 | 8,14% vs 8,17% | 0,24844 vs 0,24913
- **Istilah:** *holdout set*, AUC (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 37, 54, 55; Kaggle LB privat; reports/DECISIONS.md D011
- **Narasi pembicara (±45 detik):** Apakah skor ini bisa dipercaya? Ada tiga bukti. Pertama, data uji diambil acak per baris. Model yang dilatih untuk membedakan data latih dan data uji hanya mencapai AUC 0,4997, setara menebak acak. Karena itu validasi acak sepuluh fold meniru data uji. Kedua, komponen yang diterima dipilih di holdout set internal; leaderboard publik hanya mengonfirmasi arahnya, dan satu kali kami menolak perubahan karena leaderboard tidak mengonfirmasi. Ketiga, skor privat dari data yang tidak pernah kami lihat, 0,24844, tidak memburuk dibanding publik 0,24913. Bila model terlalu menyesuaikan diri dengan leaderboard, skor privat biasanya turun.

### Slide 8. Pendorong konsumsi (SHAP) (45 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** Rumah kecil boros gas per m², rumah besar boros listrik per rumah
- **Isi di layar:** sumbangan ke gas per m²: 5% kawasan rumah terkecil (±37 m²) vs 5% terbesar (±173 m²) | sumbangan ke listrik per rumah: 5% rumah terbesar vs 5% terkecil | Kuartil bawah vs atas: sekitar +34% dan +35%. Periode bld_era_15: EUI sekitar +7%; iklim sekitar 15%.
- **Visual:** Dua panel dependensi SHAP digambar ulang dari nilai SHAP yang sama dengan sel 47 (img/s08_dep_size.png); dua angka besar di kanan. Angka besar, sel tabel, dan label diagram: +65% | +75%
- **Istilah:** model penjelas (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 41, 45, 47, 48; bukti/cek_kebijakan.json (kontras ukuran)
- **Narasi pembicara (±45 detik):** Sekarang bagian kebijakan. Pertanyaannya: apa yang mendorong konsumsi? Kami melatih satu model penjelas per target yang hanya memakai karakteristik kawasan yang terukur, lalu membacanya dengan SHAP, yaitu pembagian setiap tebakan menjadi sumbangan tiap variabel. Ukuran rumah mendorong dua arah. Pada lima persen kawasan dengan rumah terkecil, sumbangan ke gas per meter persegi sekitar 65 persen lebih tinggi daripada lima persen rumah terbesar, kemungkinan karena dinding dan atapnya relatif luas. Untuk listrik per rumah arahnya terbalik, sekitar 75 persen. Antara kuartil bawah dan atas, selisihnya sekitar sepertiga. Ini sumbangan pada prediksi, bukan sebab-akibat.

### Slide 9. Peta trilema 330 zona (50 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** Peta 330 zona: 118 zona Q1 memuat 47% gas dan hampir separuh potensi *retrofit*
- **Isi di layar:** Potensi *retrofit* (batas atas) 7,39 TWh = 5,23% dari 141,35 TWh gas | ASUMSI 80% gas untuk pemanasan, COP 3; asumsi seragam lain hanya menskalakan angka (71% sampai 128%, sel 51). Urutan zona tahan galat prediksi, tetapi peka bila COP berbeda per iklim (C4).
- **Visual:** Peta trilema digambar ulang dari data zona sel 49 dengan label kuadran baru (img/s09_trilema.png); tabel kuadran dan kotak asumsi di kanan. Angka besar, sel tabel, dan label diagram: Kuadran | Zona | Porsi gas | *Retrofit* | Listrik | Program | Q1 | 118 | 47% | 3,65 TWh | +102% | R1 + R2 | Q2 | 47 | 12% | 1,07 TWh | +90% | R3 | Q3 | 47 | 15% | 0,86 TWh | +99% | R2 | Q4 | 118 | 27% | 1,81 TWh | +87% | standar
- **Istilah:** kuadran, kenaikan listrik (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 48, 49, 50, 51; bukti/cek_kebijakan.json
- **Narasi pembicara (±50 detik):** Supaya bisa dipakai, data latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Sumbu mendatar adalah intensitas gas; sumbu tegak adalah kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas, dengan asumsi 80 persen gas untuk pemanasan dan efisiensi pompa panas 3. Kuadran pertama berisi 118 zona dengan 47 persen konsumsi gas dan hampir separuh potensi retrofit, yang merupakan batas atas. Asumsi seragam hanya menggeser angka. Uji yang lebih sulit menunjukkan urutan zona tahan terhadap galat prediksi, tetapi berubah bila efisiensi pompa panas berbeda per iklim, jadi daftar zona adalah kandidat, bukan keputusan akhir.

### Slide 10. Rekomendasi 1: retrofit Q1 (40 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** *Retrofit* selubung lebih dulu di zona Q1, mulai dari REG_273, REG_270, dan REG_045
- **Isi di layar:** Pemerintah kota dan penyedia jasa *retrofit* | Audit energi, lalu perbaikan dinding, atap, dan jendela di kawasan rumah kecil dan bangunan periode bld_era_15, sebelum subsidi pompa panas | Gas per m² peserta turun dibanding kawasan pembanding | Rumah kecil +65% gas per m², periode bld_era_15 +7%; potensi Q1 3,65 TWh (batas atas)
- **Visual:** Batang lima zona potensi hemat terbesar (img/s10_retrofit.png); kartu tindakan siapa, apa, ukuran, dasar bukti. Angka besar, sel tabel, dan label diagram: SIAPA | APA | UKURAN | DASAR BUKTI
- **Istilah:** Selubung bangunan (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 47, 48, 49, 50
- **Narasi pembicara (±40 detik):** Rekomendasi pertama: retrofit selubung bangunan lebih dulu di zona kuadran pertama. Dasarnya temuan SHAP tadi: kawasan rumah kecil dan bangunan periode tertentu memakai gas per meter persegi paling banyak, jadi tagihannya juga paling berat. Kandidat awalnya REG_273, REG_270, dan REG_045. Pelaksananya pemerintah kota bersama penyedia jasa retrofit, dimulai dari audit energi, karena angka potensi kami adalah batas atas. Retrofit dijalankan sebelum subsidi pompa panas, supaya kebutuhan panas turun lebih dulu. Keberhasilannya diukur dari turunnya gas per meter persegi di kawasan peserta dibanding kawasan pembanding.

### Slide 11. Rekomendasi 2: gardu dulu (45 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** Kapasitas gardu dinilai sebelum subsidi pompa panas dibuka, mulai dari REG_290
- **Isi di layar:** Operator jaringan distribusi dan pemerintah kota | Nilai kapasitas gardu, jadwalkan penguatan, lalu buka subsidi bertahap dengan tarif waktu pemakaian | Tidak ada gardu melampaui kapasitas di setiap tahap subsidi | Kenaikan = gas × 0,8 / 3 / listrik saat ini; lima zona tertinggi naik 113% sampai 121%. Tambahan absolut terbesar: REG_263, 0,76 TWh
- **Visual:** Batang lima zona kenaikan tertinggi dengan garis median (img/s11_kenaikan.png); kartu tindakan. Angka besar, sel tabel, dan label diagram: SIAPA | APA | UKURAN | DASAR BUKTI
- **Istilah:** Gardu (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 50, 51; bukti/cek_kebijakan.json
- **Narasi pembicara (±45 detik):** Rekomendasi kedua: nilai kapasitas gardu sebelum subsidi pompa panas dibuka. Di lima zona tertinggi, konsumsi listrik tahunan bisa naik 113 sampai 121 persen, jauh di atas median 95 persen, karena gas di zona itu besar dibanding listriknya hari ini. Dalam tambahan absolut, zona terbesar adalah REG_263, yang juga kandidat retrofit. Urutan ini peka terhadap efisiensi pompa panas per iklim, jadi daftar ini adalah kandidat penilaian. Operator jaringan menilai gardu dengan data lokal, lalu subsidi dibuka bertahap dengan tarif berbasis waktu pemakaian. Ukurannya: tidak ada gardu yang melampaui kapasitas setelah setiap tahap.

### Slide 12. Rekomendasi 3: kawasan bergas (40 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** Insentif pompa panas untuk kawasan bergas, dimulai sebagai pilot di zona Q2
- **Isi di layar:** Pemerintah kota bersama penyedia pompa panas | Bantuan pasang pompa panas dan bongkar ketel lama sebagai pilot bergiliran; di Q1 setelah *retrofit* dan cek gardu | Zona Q2 bergas dengan adopsi rendah, misalnya REG_249 (gas jaringan 85,54%, adopsi 3,75%) | Adopsi 40% di kawasan bergas di bawah 40%, hanya 2,73% di atas 80%, padahal itu 70% kawasan | Adopsi zona peserta naik dibanding zona pembanding
- **Visual:** Dependensi SHAP adopsi terhadap porsi gas jaringan (img/s12_dep_adopt.png); kartu tindakan. Angka besar, sel tabel, dan label diagram: SIAPA | APA | DI MANA | DASAR BUKTI | UKURAN
- **Istilah:** Pilot bergiliran (penjelasan satu kalimat tertulis di kaki slide)
- **Sumber:** sel 42, 47, 48, 50; bukti/cek_kebijakan.json
- **Narasi pembicara (±40 detik):** Rekomendasi ketiga menyangkut adopsi. Tingkat adopsi pemanas listrik sekitar 40 persen di kawasan yang sedikit tersambung gas, tetapi hanya sekitar 3 persen di kawasan bergas di atas 80 persen, padahal kawasan seperti itu 70 persen dari semua kawasan. Sosiodemografi hanya menyumbang sekitar 15 persen. Karena porsi pemanas berjumlah 100 persen, ini pola, bukan tombol. Usulan kami: bantuan pasang pompa panas dan bongkar ketel lama, dimulai sebagai pilot di zona Q2 seperti REG_249 yang kenaikan listriknya lebih rendah; di Q1 setelah retrofit dan cek gardu.

### Slide 13. Keterbatasan (35 detik)
- **Bagian rubrik:** Kebijakan dari interpretabilitas (35)
- **Judul:** Tiga hal belum diketahui, dan setiap rekomendasi sudah memperhitungkannya
- **Isi di layar:** Hubungan sebab-akibat | SHAP hanya sumbangan pada prediksi. | Program dijalankan sebagai pilot dengan zona pembanding. | Beban puncak, gardu, dan COP lokal | Data hanya energi tahunan; urutan zona peka terhadap COP per iklim. | Daftar zona adalah kandidat; gardu dinilai operator dengan data lokal. | Potensi *retrofit* yang nyata | 7,39 TWh adalah batas atas; periode bangunan saja sekitar separuhnya. | Audit energi di zona pilot menentukan target penghematan.
- **Visual:** Tiga kartu: yang belum diketahui, lalu penanganannya. Angka besar, sel tabel, dan label diagram: BELUM DIKETAHUI | PENANGANAN | BELUM DIKETAHUI | PENANGANAN | BELUM DIKETAHUI | PENANGANAN
- **Sumber:** sel 48, 51; bukti/cek_kebijakan.json
- **Narasi pembicara (±35 detik):** Kami juga jujur soal batasnya. Pertama, SHAP tidak membuktikan sebab-akibat, jadi semua program kami usulkan sebagai pilot dengan zona pembanding. Kedua, data hanya memuat energi tahunan, dan urutan zona peka terhadap efisiensi pompa panas per iklim; jadi daftar zona adalah kandidat yang dinilai operator dengan data lokal. Ketiga, potensi retrofit 7,39 terawatt-jam adalah batas atas; periode bangunan saja sekitar separuhnya, dan audit energi yang menentukan angka nyatanya.

### Slide 14. Penutup (30 detik)
- **Bagian rubrik:** Penutup
- **Judul:** Layak dipilih karena akurat, jujur, dan siap dijalankan
- **Isi di layar:** skor privat, peringkat 8 | baris validasi dan latih; hasil identik di perangkat sama | kandidat berurutan, tiga program, ukuran keberhasilan | Langkah berikutnya | Minta data beban puncak, kapasitas gardu, dan COP lokal dari operator jaringan | Pilot *retrofit* di REG_273, REG_270, dan REG_045 dengan zona pembanding | Perbarui peta zona setiap tahun dengan data konsumsi terbaru | Terima kasih. Silakan ajukan pertanyaan.
- **Visual:** Slide gelap: tiga alasan dan tiga langkah berikutnya. Angka besar, sel tabel, dan label diagram: AKURAT | 0,24844 | JUJUR | Lulus uji label acak | SIAP DIJALANKAN | 330 zona | 1 | 2 | 3
- **Sumber:** Kaggle LB privat; sel 35, 49, 50
- **Narasi pembicara (±30 detik):** Jadi, mengapa solusi ini layak dipilih? Akurat: skor privat 0,24844, peringkat 8. Jujur: lulus uji kebocoran, dan hasilnya identik setiap diulang di perangkat yang sama. Siap dijalankan: 330 zona sudah berurutan sebagai kandidat, dengan tiga program dan ukuran keberhasilannya. Langkah berikutnya: meminta data beban puncak dan kapasitas gardu dari operator jaringan, lalu memulai pilot retrofit di tiga zona pertama. Terima kasih, kami siap menjawab pertanyaan.

## D. Slide cadangan (tanya jawab)

#### C1. Bobot adaptif tidak kalah di kolom mana pun pada skor jujur, terbesar di EUI
- Isi di layar: Peringkat luas A, jumlah meter M, dan tebakan p di data latih: 2×2×2 = 8 bobot pojok per model | NNLS, semua bobot ≥ 0: 224 bobot gas, 224 EUI, dan 104 listrik untuk 429.303 baris | Skor jujur | Global | Adaptif | Selisih | Gas | 0,24494 | 0,24480 | −0,000146 | EUI | 0,65023 | 0,64978 | −0,000453 | Listrik | 0,31587 | 0,31585 | −0,000027 | Rata-rata 5 target −0,000125; LB publik 0,24934 menjadi 0,24925 (sel 37)
- Pedoman jawaban (P2): model yang paling akurat di kawasan kecil belum tentu paling akurat di kawasan besar. Setiap model punya bobot di ujung kecil dan ujung besar untuk luas, meter, dan besar tebakan; kawasan di antaranya mendapat campuran. Skor jujur bobot adaptif kini dihitung ulang dengan fungsi notebook yang sama (KFold 5): ketiga kolom membaik, terbesar di EUI, rata-rata lima target sekitar minus 0,000125, sejalan dengan catatan D012 (minus 0,000124). Catatan jujur: bentuk ini dipilih di baris yang sama, jadi hitungan ulang ini bukan bukti independen; bukti independennya adalah leaderboard, minus 0,00009. Tiga simpul tidak lebih baik daripada dua.
- Sumber: sel 23, 35, 37; bukti/skor_jujur_bobot_adaptif.json; reports/DECISIONS.md D012

#### C2. Koreksi galat wilayah menggeser tebakan sedikit mengikuti galat tetangga
- Isi di layar: Fitur koreksi (gas) | λ | rata-rata galat zona | 0,453 | rata-rata galat petak iklim | 0,346 | rata-rata galat ±30 nomor subzona | −0,020 | kNN full, W 100, k 20 | 0,037 | kNN era, W 30, k 20 | 0,126 | kNN fc, W 300, k 50 | 0,079 | kNN full, W 10, k 10 | 0,072 | kNN era, W 300, k 50 | 0,124 | kNN full, W 3000, k 200 | 0,156 | Galat latih dirata-rata per zona, petak iklim, ±30 nomor subzona, dan k kawasan termirip | Dikecilkan n/(n + 30) atau n/(n + 5) bila tetangga sedikit | λ dipilih di *holdout set* 30%, lalu dibekukan di resid_knn.json | Besar koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195 (sel 35) | Koreksi kNN menurunkan LB sekitar 0,0004 (0,2499 menjadi 0,24954); koreksi tingkat grup sudah ada sejak v12 | λ negatif kecil muncul karena fitur tetangga saling berkorelasi; yang divalidasi adalah efek totalnya. Versi ridge diuji di audit akhir dan tidak lebih baik secara konsisten.
- Pedoman jawaban (P3): setelah digabung, galat masih berkelompok; kalau tetangga yang mirip sama-sama ditebak terlalu rendah, kawasan ini kemungkinan juga terlalu rendah. Target data uji tidak pernah dipakai; λ dipilih di holdout yang tidak ikut menyusun tetangga. Konstanta K, WIN, dan KSHR diperiksa ulang di audit akhir (D015). Bila ditanya langkah v12 ke v15 di tabel sel 37: langkah itu menggabungkan beberapa perubahan; koreksi kNN sendiri sekitar minus 0,0004 di leaderboard.
- Sumber: sel 22, 35, 37; reports/DECISIONS.md D015; reports/JOURNAL.md 15 sampai 16 Okt dan audit akhir

#### C3. Penyesuaian akhir menangani dua batas data, dan hasilnya identik setiap diulang di perangkat sama
- Isi di layar: Langkah | Gas | EUI | Listrik | S1: kawasan 2 sampai 3 meter | +20,1% | +18,4% | +18,7% | S1: kawasan 4 meter | +6,4% | +3,2% | +7,1% | S3: faktor k | 0,505 | 0,657 | tidak | S3: a0 dan b0 | 0,0081; 0,393 | 0,0084; 0,387 | tidak | S4: batas | batas latih | batas latih | batas latih | S1: kawasan 2 sampai 3 meter rata-rata ditebak sekitar 20% terlalu rendah oleh semua keluarga model | S3: tebakan gas dekat batas 999.765 kWh ditarik turun (normal terpotong) | S4: dipotong ke batas data latih; 77 kawasan duplikat disalin | Asal: S1 dari prediksi OOF penuh, k pada S3 dari *holdout*, batas S4 dari data latih | LB 0,24920 menjadi 0,24913; perkiraan *holdout* sama | Reprodusibilitas: *seed* dikunci, mode deterministik, *thread* BLAS dibatasi satu. Sidik sha256 aec745ab…d66e tercetak di sel 35, sama dengan berkas yang dipilih (perangkat sama).
- Pedoman jawaban (P5): assemble menjalankan urutan tetap: gabung, koreksi, sesuaikan batas, salin duplikat, periksa, tulis. Yang di-fit di dalamnya hanya bobot gabungan. S1 berasal dari prediksi out-of-fold penuh, k pada S3 dari holdout, dan batas S4 dari data latih; semuanya dibekukan di satu berkas konfigurasi (D015). Efeknya 0,00007 di leaderboard, sama dengan perkiraan holdout.
- Sumber: sel 22, 35, 37; reports/DECISIONS.md D015

#### C4. Urutan zona tahan galat prediksi, tetapi peka terhadap COP yang berbeda per iklim
- Isi di layar: Porsi gas untuk pemanasan | COP 2,5 | COP 3 | COP 3,5 | 70% | +99,6% | +83,0% | +71,1% | 80% | +113,8% | +94,8% | +81,3% | 90% | +128,0% | +106,7% | +91,5% | Rumus: gas × porsi pemanasan / COP / listrik. Asumsi seragam mengalikan semua zona dengan faktor yang sama, jadi urutan zona tetap secara aljabar, bukan karena diuji. | Median 94,85% (sel 50) dikali (porsi / 0,8) × (3 / COP); *writeup* Tabel 4 memuat nilai yang sama dalam bilangan bulat (+71% sampai +128%). | UJI KETAHANAN URUTAN ZONA | **Galat prediksi** baris uji, 200 simulasi: Spearman 0,998; median 8 zona pindah kuadran; 5 zona teratas tetap. | **COP 2,5 sampai 3,5 menurut iklim zona:** Spearman 0,81; 68 zona pindah kuadran; 1 dari 5 zona teratas tetap.
- Pedoman jawaban: asumsi seragam hanya menskalakan semua zona, jadi tidak menguji apa pun. Kami menguji dua hal yang bisa mengubah urutan. Galat prediksi pada baris uji hampir tidak berpengaruh karena zona merata-ratakan banyak kawasan. COP yang berbeda per iklim berpengaruh besar, sehingga daftar zona kami sebut kandidat dan operator menilai dengan data lokal. Beban puncak bisa jauh lebih besar karena pemanasan menumpuk di bulan terdingin dan COP turun saat dingin.
- Sumber: sel 50, 51; bukti/cek_kebijakan.json; writeup Tabel 4

#### C5. Validasi per zona keliru untuk data ini karena data uji diambil acak per baris
- Isi di layar: Skema | Rata-rata wilayah | Gas | EUI | Listrik | acak per baris | tanpa | 0,2573 | 0,6721 | 0,3262 | acak per baris | dengan | 0,2562 | 0,6699 | 0,3255 | per zona | tanpa | 0,2708 | 0,6923 | 0,3302 | per zona | dengan | 0,3637 | 0,8331 | 0,3474 | Data uji diambil acak per baris (AUC 0,4997; subzona baru 8,14% vs 8,17%). Skema per zona menguji situasi yang tidak terjadi di data uji dan membuat fitur rata-rata wilayah tampak merusak. | Pada skema yang sesuai data uji, fitur yang sama menurunkan galat di ketiga target. Skema yang keliru akan membuang fitur yang berguna.
- Pedoman jawaban: untuk data spasial, pembagian per wilayah sering dianjurkan. Namun 92% kawasan uji berada di subzona yang juga ada di data latih, dan validasi adversarial tidak bisa membedakan latih dan uji. Jadi validasi acak per baris adalah skema yang meniru data uji.
- Sumber: writeup Tabel 1; sel 54, 55

#### C6. Adopsi di subzona baru dimodelkan sebagai peluang dari jumlah rumah tangga
- Isi di layar: Kolom komposisi pemanas adalah pecahan jumlah rumah tangga N. Bilangan bulat terkecil yang membuat semua pecahan bulat adalah N. | K = rumah tangga di luar kategori pemanas yang tercatat. Model menebak peluang rumah tangga K bukan pengguna pemanas listrik. | Dua LightGBM (q_xent, r_reg) dirata-rata; validasi 5 *fold* per subzona, dibobot ulang ke komposisi subzona baru. | 1,03 poin | RMSE pada subzona baru (0,143 setelah dibagi σ adopsi) | 8,14% | kawasan uji yang memakai model ini; 91,86% sisanya disalin
- Pedoman jawaban: adopsi seragam per subzona, jadi kawasan di subzona yang dikenal cukup disalin. Untuk subzona baru, struktur cacah rumah tangga dipakai: N dari pecahan komposisi pemanas, K rumah tangga di luar kategori tercatat. Galatnya dekat batas derau binomial.
- Sumber: sel 28, 35, 39

#### C7. Uji label acak: fitur bertarget tidak berubah saat label baris yang dinilai diacak
- Isi di layar: Blok fitur (fold 0 dari 10) | Kolom | Baris | Berubah | kNN-target, baris validasi | 39 | 42.931 | 0 | kNN-target, data uji | 39 | 183.987 | 0 | rata-rata wilayah, validasi | 42 | 42.931 | 0 | rata-rata wilayah, data uji | 42 | 183.987 | 0 | kNN-target, baris latih bagian 0 | 39 | 77.275 | 0 | rata-rata wilayah, baris latih bagian 0 | 42 | 77.275 | 0 | KONTROL: tetangga termasuk validasi | 15 | 42.931 | 42.931 | Label kelima target diacak (1) di baris validasi dan (2) di satu bagian dalam baris latih; fitur dihitung ulang dengan modul notebook yang sama. | Lulus bila blok yang tidak boleh membaca label itu identik bit, dan kontrol bocor berubah. Cakupan: satu fold, kNN-target dan rata-rata wilayah. | Bobot dari prediksi *out-of-fold* (0,25096) setara dengan bobot yang di-fit di *holdout set* (0,25095); ensemble pengembangan 13 Okt (D002).
- Pedoman jawaban (P1 dan P4): untuk fitur kNN-target dan rata-rata wilayah, baris validasi dan uji tidak membaca label validasi, dan baris latih tidak membaca label bagian dalamnya sendiri; keduanya diuji dengan label acak. Cakupannya satu fold; koreksi galat wilayah hanya memakai galat baris latih untuk baris uji. Untuk pergeseran bobot (P1): di holdout, prediksi berasal dari rata-rata model fold seperti data uji, dan bobot dari prediksi out-of-fold memberi skor setara dengan bobot yang di-fit di holdout (bukti pengembangan, ensemble 13 Okt). Catatan jujur: XGBoost dan LightGBM memakai early stopping pada fold validasinya sendiri (sel 36), sehingga galat out-of-fold sedikit optimistis; keputusan diambil dari holdout.
- Sumber: bukti/uji_kebocoran_*.json; reports/DECISIONS.md D002; sel 36

#### C8. SHAP dibaca dari model penjelas terpisah karena gabungan 41 model memakai fitur yang bukan tuas kebijakan
- Isi di layar: Model penjelas | R² | log intensitas gas (EUI) | 0,563 | log gas per meter | 0,747 | log listrik per meter | 0,491 | adopsi pemanas listrik | 0,991 | R² adopsi tinggi karena porsi jenis pemanas berjumlah 100%, bukan karena perilaku terjelaskan (identitas komposisi). | Gabungan 41 model memakai rata-rata target wilayah, yang bukan tuas kebijakan, dan sulit dibaca. | Model penjelas hanya memakai karakteristik terukur: bangunan, periode, komposisi pemanas, sosiodemografi, iklim, agregat kawasan. | Porsi kontribusi (dibaca dari grafik sel 43): adopsi, komposisi pemanas sekitar 81% dan sosiodemografi sekitar 15%; listrik per meter, sosiodemografi sekitar 40%.
- Pedoman jawaban: kecocokan model penjelas dilaporkan supaya batas keyakinannya terlihat. SHAP dihitung dengan TreeSHAP eksak pada sampel 20.000 baris latih untuk tabel dan grafik; peta zona memakai kontribusi pada seluruh 613 ribu kawasan.
- Sumber: sel 40, 41, 42, 43, 45

## E. Daftar periksa arahan (bagian F)

| Butir arahan | Status | Bukti |
|---|---|---|
| Pesan utama dapat dipahami dengan cepat | terpenuhi | slide 1 menyatakan kesimpulan dan tiga rekomendasi dalam 30 detik pertama |
| Alur membentuk argumen yang logis | terpenuhi | masalah (2), data (3), fitur (4, 5), model (6), kepercayaan (7), pendorong (8), peta (9), tindakan (10 sampai 12), batas (13), alasan memilih (14) |
| Setiap judul slide menyampaikan kesimpulan | terpenuhi | semua judul berupa kalimat pernyataan; uji: membaca 14 judul saja sudah menceritakan argumen |
| Setiap slide punya satu pesan utama | terpenuhi | rekomendasi dipecah menjadi tiga slide; rincian arsitektur pindah ke cadangan |
| Data penting relevan dan bersumber (nomor sel) | terpenuhi | setiap angka penting punya label sumber; `check_deck.py` memeriksa angka terhadap lembar fakta (bagian F) |
| Masalah, model, dan rekomendasi terhubung | terpenuhi | rekomendasi menyebut temuan SHAP dan angka peta di baris "Dasar bukti" |
| Fakta, asumsi, dan interpretasi dibedakan | terpenuhi | label FAKTA, HASIL, ASUMSI, INTERPRETASI |
| Keterbatasan dan risiko diakui | terpenuhi | slide 13 (sebab-akibat, COP lokal dan beban puncak, potensi *retrofit* sebagai batas atas); label asumsi di slide 2, 9, 11 |
| Istilah teknis dijelaskan | terpenuhi | baris "Istilah" di kaki slide 1 sampai 12 |
| Desain konsisten dan terbaca | terpenuhi di pratinjau | posisi judul, label, dan kaki seragam; ukuran minimal 11 pt hanya untuk label kecil |
| Isi sesuai bobot rubrik dan batas 10 menit | terpenuhi | bagian B; total 9 menit 20 detik |
| Penutup memberi alasan kuat dan langkah berikutnya | terpenuhi | slide 14 |
| Sudah dilatih dengan pengatur waktu | **belum, tugas tim** | `07_kartu_latihan.md` menyediakan kolom latihan 1 dan 2 |
