# Kartu Latihan: Deck Final v2 (10 menit, 3 orang)

Dibuat otomatis oleh `build_deck_v2.py` dari narasi slide yang sama dengan PPTX. Kecepatan acuan 130 kata per menit. Total rencana 9 menit 20 detik, menyisakan 40 detik untuk serah terima dan jeda. Latih minimal dua kali penuh dengan pengatur waktu, lalu catat waktu tiap slide.

## A. Peran tim

**Orang 1 · Jangkar**: slide 1, 2, 14 (1 menit 35 detik). Membuka dan menutup presentasi. Selama Orang 2 dan Orang 3 berbicara, ia berbagi layar dan memindahkan slide, memegang pengatur waktu, dan menyiapkan slide cadangan. Saat tanya jawab ia mengarahkan pertanyaan. Pilih anggota yang paling lancar berbicara dan memahami cerita dari awal sampai akhir.

**Orang 2 · Teknis**: slide 3, 4, 5, 6, 7 (3 menit 30 detik). Menyampaikan data, fitur, arsitektur, dan validasi, lalu menjawab pertanyaan teknis. Pilih anggota yang paling paham model dan validasi.

**Orang 3 · Kebijakan**: slide 8, 9, 10, 11, 12, 13 (4 menit 15 detik). Menyampaikan SHAP, peta zona, tiga rekomendasi, dan keterbatasan, lalu menjawab pertanyaan kebijakan. Pilih anggota yang paling nyaman menjelaskan hasil kepada pembuat kebijakan.

## B. Urutan bicara dan serah terima

- **0:00** Orang 1: slide 1 dan 2
- **1:05** Orang 2: slide 3 sampai 7
- **4:35** Orang 3: slide 8 sampai 13
- **8:50** Orang 1: slide 14
- **9:20** tanya jawab, dipandu Orang 1

Kalimat serah terima, diucapkan di akhir slide sebelum pergantian (sekitar 3 detik):

- Akhir slide 2: "Untuk menjawabnya, [nama Orang 2] akan memulai dari datanya."
- Akhir slide 7: "Dengan model yang sudah teruji itu, [nama Orang 3] akan masuk ke bagian kebijakan."
- Akhir slide 13: "[nama Orang 1] akan menutup presentasi kami."

Orang 1 memindahkan slide tepat setelah kalimat terakhir tiap slide. Kalimat terakhir itu tertulis di naskah bagian C, jadi Orang 1 cukup mengikuti naskah.

## C. Naskah per orang

### Orang 1 · Jangkar

#### Slide 1 · Pesan utama (30 detik, selesai di 0:30)

Selamat pagi, Bapak dan Ibu juri. Kami tim BISMILLAH, dan kami mulai dari kesimpulan. Dari lima besaran energi untuk 613 ribu kawasan, 184 ribu di antaranya kami prediksi, kami menyusun urutan program untuk 330 zona: retrofit dulu di zona yang boros gas, kapasitas gardu dinilai sebelum subsidi pompa panas, dan pilot pompa panas di kawasan bergas. Model di baliknya meraih peringkat 8 di leaderboard privat dan lulus uji kebocoran.

**Angka wajib:** 613, 184, 330

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 2 · Masalah (35 detik, selesai di 1:05)

Mengapa masalah ini layak diselesaikan? Kota yang beralih dari pemanas gas ke pemanas listrik harus memenuhi tiga tuntutan sekaligus: emisi turun, jaringan listrik tetap andal, dan biaya tetap terjangkau. Kelima target lomba mengukur ketiganya. Hasil kami memperlihatkan risikonya. Bila seluruh pemanas gas diganti pompa panas, konsumsi listrik tahunan zona naik dengan median hampir 95 persen. Jadi pertanyaannya bukan apakah kota beralih, tetapi zona mana yang didahulukan dan dengan program apa. *Untuk menjawabnya, [nama Orang 2] akan memulai dari datanya.*

**Angka wajib:** 95 persen

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 14 · Penutup (30 detik, selesai di 9:20)

Jadi, mengapa solusi ini layak dipilih? Akurat: skor privat 0,24844, peringkat 8. Jujur: lulus uji kebocoran, dan hasilnya identik setiap diulang di perangkat yang sama. Siap dijalankan: 330 zona sudah berurutan sebagai kandidat, dengan tiga program dan ukuran keberhasilannya. Langkah berikutnya: meminta data beban puncak dan kapasitas gardu dari operator jaringan, lalu memulai pilot retrofit di tiga zona pertama. Terima kasih, kami siap menjawab pertanyaan.

**Angka wajib:** 0,24844, 330

Latihan 1: ........ detik. Latihan 2: ........ detik.

### Orang 2 · Teknis

#### Slide 3 · Struktur data (35 detik, selesai di 1:40)

Jawaban pertama datang dari struktur data, sebelum model dilatih. Pemeriksaan di sel 25 menemukan tiga hal. Gas selalu sama dengan EUI dikali luas lantai, dengan galat setara presisi mesin, jadi satu keluarga model cukup untuk dua kolom. Porsi bangunan bersih tepat sama dengan satu rumus di seluruh baris latih, jadi dihitung, bukan ditebak. Adopsi pemanas listrik seragam dalam satu subzona, dan 91,86 persen kawasan uji berada di subzona yang sudah dikenal. Model besar hanya dibutuhkan untuk gas, EUI, dan listrik.

**Angka wajib:** 91,86 persen

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 4 · Fitur ukuran rumah (35 detik, selesai di 2:15)

Fitur fisik terkuat kami sangat sederhana: luas lantai dibagi jumlah meteran, kira-kira luas rata-rata satu rumah. Gas total sebuah kawasan terutama ditentukan oleh banyaknya rumah, sedangkan ukuran rumah menentukan cara energi dipakai. Pada model penjelas, fitur ini peringkat pertama untuk intensitas gas dan untuk listrik per rumah, hampir tiga kali fitur kedua. Fitur ini tidak mungkin membocorkan jawaban, karena hanya membagi dua kolom input yang juga tersedia di data uji, baris demi baris.

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 5 · Fitur kawasan termirip (45 detik, selesai di 3:00)

Fitur kedua bertanya: berapa konsumsi kawasan lain yang stok bangunannya paling mirip? Pencarian dibatasi pada jendela nomor subzona, karena nomor subzona ternyata berurutan secara geografis: 99,07 persen baris berurutan berada di zona yang sama. Fitur ini memakai target tetangga, jadi kami menghitungnya out-of-fold dua lapis, seperti skema di kiri. Lalu kami mengujinya: label baris validasi kami acak, lalu label satu bagian dalam baris latih. Fitur yang tidak boleh membaca label itu tetap identik sampai bit terakhir, sedangkan versi yang sengaja dibuat bocor langsung berubah. Jadi uji ini mampu menangkap kebocoran, dan fitur kami lolos.

**Angka wajib:** 99,07 persen

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 6 · Arsitektur tiga jalur (50 detik, selesai di 3:50)

Karena metrik menilai setiap kolom terpisah, setiap target boleh memakai cara terbaiknya. Jalur pertama untuk gas dan EUI: 28 model, karena keduanya satu besaran. Jalur kedua untuk listrik: 13 model, karena pendorongnya berbeda. Jalur ketiga hampir tanpa model: porsi bangunan bersih dihitung dengan rumus, adopsi disalin dari subzonanya, dan dua LightGBM hanya untuk 8 persen kawasan di subzona baru. Model di dua jalur pertama beragam, lalu digabung, dikoreksi, dan disesuaikan dengan batas data. Angka di kanan adalah galat jujur gabungan sebelum koreksi; koreksi dan penyesuaian menurunkan skor lebih jauh. Notebook-nya 56 sel dengan satu sakelar, dan hasilnya identik byte setiap diulang di perangkat yang sama.

**Angka wajib:** 28, 13, 8 persen, 56

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 7 · Teruji di data tersembunyi (45 detik, selesai di 4:35)

Apakah skor ini bisa dipercaya? Ada tiga bukti. Pertama, data uji diambil acak per baris. Model yang dilatih untuk membedakan data latih dan data uji hanya mencapai AUC 0,4997, setara menebak acak. Karena itu validasi acak sepuluh fold meniru data uji. Kedua, komponen yang diterima dipilih di holdout set internal; leaderboard publik hanya mengonfirmasi arahnya, dan satu kali kami menolak perubahan karena leaderboard tidak mengonfirmasi. Ketiga, skor privat dari data yang tidak pernah kami lihat, 0,24844, tidak memburuk dibanding publik 0,24913. Bila model terlalu menyesuaikan diri dengan leaderboard, skor privat biasanya turun. *Dengan model yang sudah teruji itu, [nama Orang 3] akan masuk ke bagian kebijakan.*

**Angka wajib:** 0,4997, 0,24844, 0,24913

Latihan 1: ........ detik. Latihan 2: ........ detik.

### Orang 3 · Kebijakan

#### Slide 8 · Pendorong konsumsi (SHAP) (45 detik, selesai di 5:20)

Sekarang bagian kebijakan. Pertanyaannya: apa yang mendorong konsumsi? Kami melatih satu model penjelas per target yang hanya memakai karakteristik kawasan yang terukur, lalu membacanya dengan SHAP, yaitu pembagian setiap tebakan menjadi sumbangan tiap variabel. Ukuran rumah mendorong dua arah. Pada lima persen kawasan dengan rumah terkecil, sumbangan ke gas per meter persegi sekitar 65 persen lebih tinggi daripada lima persen rumah terbesar, kemungkinan karena dinding dan atapnya relatif luas. Untuk listrik per rumah arahnya terbalik, sekitar 75 persen. Antara kuartil bawah dan atas, selisihnya sekitar sepertiga. Ini sumbangan pada prediksi, bukan sebab-akibat.

**Angka wajib:** 65 persen, 75 persen

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 9 · Peta trilema 330 zona (50 detik, selesai di 6:10)

Supaya bisa dipakai, data latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Sumbu mendatar adalah intensitas gas; sumbu tegak adalah kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas, dengan asumsi 80 persen gas untuk pemanasan dan efisiensi pompa panas 3. Kuadran pertama berisi 118 zona dengan 47 persen konsumsi gas dan hampir separuh potensi retrofit, yang merupakan batas atas. Asumsi seragam hanya menggeser angka. Uji yang lebih sulit menunjukkan urutan zona tahan terhadap galat prediksi, tetapi berubah bila efisiensi pompa panas berbeda per iklim, jadi daftar zona adalah kandidat, bukan keputusan akhir.

**Angka wajib:** 330, 80 persen, 118, 47 persen

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 10 · Rekomendasi 1: retrofit Q1 (40 detik, selesai di 6:50)

Rekomendasi pertama: retrofit selubung bangunan lebih dulu di zona kuadran pertama. Dasarnya temuan SHAP tadi: kawasan rumah kecil dan bangunan periode tertentu memakai gas per meter persegi paling banyak, jadi tagihannya juga paling berat. Kandidat awalnya REG_273, REG_270, dan REG_045. Pelaksananya pemerintah kota bersama penyedia jasa retrofit, dimulai dari audit energi, karena angka potensi kami adalah batas atas. Retrofit dijalankan sebelum subsidi pompa panas, supaya kebutuhan panas turun lebih dulu. Keberhasilannya diukur dari turunnya gas per meter persegi di kawasan peserta dibanding kawasan pembanding.

**Angka wajib:** REG_273, REG_270, REG_045

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 11 · Rekomendasi 2: gardu dulu (45 detik, selesai di 7:35)

Rekomendasi kedua: nilai kapasitas gardu sebelum subsidi pompa panas dibuka. Di lima zona tertinggi, konsumsi listrik tahunan bisa naik 113 sampai 121 persen, jauh di atas median 95 persen, karena gas di zona itu besar dibanding listriknya hari ini. Dalam tambahan absolut, zona terbesar adalah REG_263, yang juga kandidat retrofit. Urutan ini peka terhadap efisiensi pompa panas per iklim, jadi daftar ini adalah kandidat penilaian. Operator jaringan menilai gardu dengan data lokal, lalu subsidi dibuka bertahap dengan tarif berbasis waktu pemakaian. Ukurannya: tidak ada gardu yang melampaui kapasitas setelah setiap tahap.

**Angka wajib:** 113, 121 persen, 95 persen, REG_263

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 12 · Rekomendasi 3: kawasan bergas (40 detik, selesai di 8:15)

Rekomendasi ketiga menyangkut adopsi. Tingkat adopsi pemanas listrik sekitar 40 persen di kawasan yang sedikit tersambung gas, tetapi hanya sekitar 3 persen di kawasan bergas di atas 80 persen, padahal kawasan seperti itu 70 persen dari semua kawasan. Sosiodemografi hanya menyumbang sekitar 15 persen. Karena porsi pemanas berjumlah 100 persen, ini pola, bukan tombol. Usulan kami: bantuan pasang pompa panas dan bongkar ketel lama, dimulai sebagai pilot di zona Q2 seperti REG_249 yang kenaikan listriknya lebih rendah; di Q1 setelah retrofit dan cek gardu.

**Angka wajib:** 40 persen, 3 persen, 80 persen, 70 persen, 15 persen, 100 persen, REG_249

Latihan 1: ........ detik. Latihan 2: ........ detik.

#### Slide 13 · Keterbatasan (35 detik, selesai di 8:50)

Kami juga jujur soal batasnya. Pertama, SHAP tidak membuktikan sebab-akibat, jadi semua program kami usulkan sebagai pilot dengan zona pembanding. Kedua, data hanya memuat energi tahunan, dan urutan zona peka terhadap efisiensi pompa panas per iklim; jadi daftar zona adalah kandidat yang dinilai operator dengan data lokal. Ketiga, potensi retrofit 7,39 terawatt-jam adalah batas atas; periode bangunan saja sekitar separuhnya, dan audit energi yang menentukan angka nyatanya. *[nama Orang 1] akan menutup presentasi kami.*

**Angka wajib:** 7,39

Latihan 1: ........ detik. Latihan 2: ........ detik.

## D. Tanya jawab (10 menit)

1. Juri bertanya. Orang 1 mengulang inti pertanyaan dalam satu kalimat dan menyebut siapa yang menjawab. Pengulangan ini memberi waktu berpikir dan mencegah dua orang bicara bersamaan di Zoom.
2. Penjawab memakai urutan inti, bukti (angka dan nomor slide atau sel), lalu batasan, maksimal sekitar 45 detik.
3. Bila perlu bukti, Orang 1 membuka slide cadangan. Di mode slideshow PowerPoint, ketik nomor slide lalu Enter: C1 = 15, C2 = 16, C3 = 17, C4 = 18, C5 = 19, C6 = 20, C7 = 21, C8 = 22.
4. Anggota lain tidak memotong. Bila ingin menambah, tunggu jawaban selesai lalu tambahkan satu kalimat.
5. Bila tidak tahu, katakan terus terang, sebut yang sudah diketahui, lalu jadikan langkah berikutnya.

**Pembagian penjawab:**

- **Orang 2:** 06 nomor 1 sampai 7 dan 13; cadangan C1, C2, C3, C5, C7.
- **Orang 3:** 06 nomor 8 sampai 12 dan 14; cadangan C4, C6, C8.
- **Orang 1:** pertanyaan umum dan peran AI (06 bagian C). Ia juga menengahi bila jawaban terlalu panjang.

## E. Daftar periksa Zoom

- Hanya Orang 1 yang berbagi layar, dari PowerPoint mode presenter, sepanjang presentasi. Orang 2 menyiapkan PPTX dan PDF yang sama dan siap mengambil alih bila koneksi Orang 1 putus.
- Nama tampilan: "BISMILLAH · Nama · peran". Kamera menyala, latar netral, mikrofon dibisukan saat tidak bicara.
- Grup chat privat tim dipakai untuk mengirim angka cepat saat tanya jawab; 06 terbuka di HP masing-masing.
- Uji audio dan berbagi layar 15 menit sebelum mulai, termasuk satu kali pindah slide.

## F. Alternatif dua pembicara

Bila satu anggota tidak ingin memaparkan: Orang 2 membawakan slide 1 sampai 7 dan Orang 3 slide 8 sampai 14. Orang 1 tetap menjadi operator slide, pemegang waktu, dan pengarah tanya jawab, serta menjawab pertanyaan umum. Serah terima hanya satu kali (slide 8). Kekurangannya, juri melihat satu anggota tidak ikut memaparkan. Bila panduan lomba mewajibkan semua anggota berbicara, pakai pembagian tiga orang.

