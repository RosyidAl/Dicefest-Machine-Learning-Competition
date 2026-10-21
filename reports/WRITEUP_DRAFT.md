# Memahami Data Dulu, Baru Memodelkan: Prediksi Trilema Energi Perkotaan dan Peta Kebijakan per Zona

DICE FEST 2026, Machine Learning. Tim BISMILLAH

## Ringkasan
Tugas kompetisi ini adalah memperkirakan lima angka energi untuk setiap kawasan permukiman: konsumsi gas, konsumsi listrik, intensitas gas per meter persegi, porsi bangunan bersih, dan porsi rumah tangga yang sudah memakai pemanas listrik. Ada 429 ribu kawasan untuk belajar dan 184 ribu kawasan yang harus ditebak.

Sebelum melatih model apa pun, kami membedah datanya. Ternyata satu dari lima target bisa dihitung tepat dari kolom yang tersedia, satu lagi sudah diketahui untuk 92% kawasan uji, dan dua target lain terikat rumus fisika sederhana. Temuan ini membuat pekerjaan kami bisa dipusatkan ke bagian yang memang sulit.

Bagian sulit itu kami kerjakan dengan tiga hal: fitur yang meniru cara energi benar-benar dipakai (ukuran rumah, umur bangunan, kemiripan dengan kawasan tetangga), gabungan 41 model yang saling menutup kelemahan, dan pengujian yang jujur sehingga setiap perbaikan di validasi kami juga muncul di leaderboard. Skor akhir kami di leaderboard publik 0,24913.

Untuk sisi kebijakan, kami membuat model penjelas yang hanya memakai karakteristik kawasan yang bisa diukur, lalu menyusun peta prioritas untuk 330 zona: mana yang perlu perbaikan bangunan dulu, mana yang siap beralih ke pompa panas listrik (heat pump), dan mana yang jaringan listriknya harus diperkuat sebelum elektrifikasi.

## 1. Masalahnya
Kota yang ingin lepas dari pemanas berbahan gas menghadapi tiga tarikan sekaligus. Emisi harus turun, jaringan listrik harus tetap kuat ketika pemanas pindah ke listrik, dan biaya energi rumah tangga tidak boleh melonjak. Kelima target kompetisi memotret ketiga sisi itu:

| Target | Artinya | Sisi trilema |
|---|---|---|
| gas total (kWh) | konsumsi gas setahun di kawasan | emisi |
| listrik total (kWh) | konsumsi listrik setahun | beban jaringan |
| EUI (kWh/m²) | gas per meter persegi lantai, ukuran boros tidaknya bangunan | emisi dan biaya |
| porsi bangunan bersih (%) | kualitas stok bangunan | emisi |
| adopsi pemanas listrik (%) | seberapa jauh transisi sudah berjalan | transisi |

Satu baris data adalah satu kawasan kecil (kira-kira setara satu kode pos) berisi rata-rata 20 rumah tangga. Fiturnya ada 201 kolom yang namanya disamarkan panitia: dimensi bangunan, era pembangunan, campuran jenis pemanas, iklim, dan sosio-demografi, ditambah dua kode lokasi (330 zona dan sekitar 150 ribu subzona).

Skor dihitung sebagai rata-rata dari lima galat, masing-masing dibagi sebaran targetnya (RMSE/sigma). Tebakan rata-rata yang sama untuk semua kawasan menghasilkan sekitar 1,0. Karena tiap target dinilai terpisah, tiap target boleh punya cara pengerjaan sendiri. Pada model akhir kami, EUI menyumbang sekitar separuh total galat, listrik seperempat, gas seperlima, dan adopsi sisanya. Fokus kerja kami mengikuti porsi itu.

## 2. Apa yang kami temukan di data
### 2.1 Dua rumus fisika tersembunyi
EUI dikali kolom `bld_dim_06` selalu sama dengan gas total, sampai digit terakhir komputer (selisih terbesar 7 per 10 ribu triliun). Artinya `bld_dim_06` adalah luas lantai, dan gas serta EUI sebenarnya satu besaran yang dilihat dari dua sisi. Porsi bangunan bersih juga selalu sama dengan 100 x `bld_dim_02` / `bld_dim_01`, dibulatkan dua desimal, di seluruh 429 ribu baris. Target ini cukup dihitung, tidak perlu ditebak.

### 2.2 Adopsi sama untuk satu subzona
Nilai adopsi pemanas listrik tidak pernah berbeda di dalam satu subzona (subzona adalah unit sensus). Karena 92% kawasan uji berada di subzona yang juga muncul di data latih, nilainya tinggal disalin. Untuk 8% sisanya, kami memanfaatkan satu sifat menarik: kolom campuran jenis pemanas adalah pecahan cacah rumah tangga. Bilangan bulat terkecil yang membuat semua pecahan itu menjadi utuh adalah jumlah rumah tangga di subzona. Dari situ adopsi bisa dimodelkan sebagai "berapa rumah dari K rumah yang memakai listrik", persis seperti menghitung peluang.

### 2.3 Sebagian galat memang tidak bisa dihindari
Konsumsi energi satu rumah bisa sangat berbeda dari tetangganya karena hal yang tidak ada di data (jumlah penghuni, kebiasaan, termostat). Ketika konsumsi beberapa rumah dijumlahkan, perbedaan itu sebagian saling menutup, dan makin banyak meter di kawasan, makin kecil sisanya. Pola ini terlihat jelas di data (Gambar 1): sebaran galat mengikuti a + b/M, dengan M jumlah meter. Sekitar tiga perempat galat gas dan listrik berasal dari variasi antar rumah ini. Jadi model yang lebih besar tidak akan banyak membantu; yang membantu adalah mengurangi kesalahan acak model itu sendiri.

![Gambar 1. Sebaran galat model per kelompok jumlah meter. Titik mengikuti kurva a + b/M: makin banyak rumah di kawasan, makin kecil galat. Garis putus-putus adalah bagian yang masih bisa diperbaiki model.](figures/fig2_noise_floor.png)

### 2.4 Batas di data
Gas di data latih tidak pernah melebihi 999.765 kWh, seolah dipotong tepat di bawah satu juta. Kawasan dengan sangat sedikit meter (2 sampai 4) juga berperilaku lain dari yang lain. Keduanya kami tangani di tahap akhir (bagian 5.4).

## 3. Cara kami menguji model
Model yang bagus di data latih belum tentu bagus di data uji. Karena itu kami harus meniru cara panitia memisahkan data. Kami memeriksa beberapa hal: data uji tepat 30% dari total, kawasan uji tersebar merata di semua zona, porsi kawasan uji di subzona baru 8,14% (sama dengan yang diharapkan dari pengacakan, 8,17%), dan sebuah model yang dilatih untuk membedakan baris latih dari baris uji tidak lebih baik dari menebak acak. Kesimpulannya, data uji diambil acak per baris.

Dengan begitu, cara menguji yang tepat adalah membagi data latih secara acak menjadi 10 bagian, melatih di sembilan bagian, menguji di satu bagian, lalu bergiliran. Pilihan ini penting. Kalau kami membagi data per zona (seperti banyak tutorial sarankan untuk data spasial), fitur lokasi terlihat merusak model. Padahal di data uji yang sebenarnya, fitur yang sama justru termasuk yang paling berguna (Tabel 1).

**Tabel 1.** Galat LightGBM (makin kecil makin baik) dengan dan tanpa fitur rata-rata wilayah, pada dua cara pembagian data.

| Cara membagi data | Fitur rata-rata wilayah | Gas | EUI | Listrik |
|---|---|---|---|---|
| acak per baris (sama dengan data uji) | tanpa | 0,2573 | 0,6721 | 0,3262 |
| acak per baris (sama dengan data uji) | dengan | 0,2562 | 0,6699 | 0,3255 |
| per zona | tanpa | 0,2708 | 0,6923 | 0,3302 |
| per zona | dengan | 0,3637 | 0,8331 | 0,3474 |

Untuk langkah-langkah setelah model digabung (koreksi dan penyesuaian akhir), kami menyisihkan 30% data latih sebagai "ujian internal": semua model dilatih ulang tanpa bagian ini, lalu setiap ide baru diuji di sana. Ujian internal ini ternyata sangat akurat. Setiap kali ia memperkirakan skor membaik sekian, leaderboard membaik hampir persis sebesar itu (Gambar 2). Karena itu kami tidak pernah memilih perubahan berdasarkan leaderboard publik, yang hanya memakai 30% data uji.

![Gambar 2. Perkembangan skor leaderboard publik. Setiap langkah dipilih dari ujian internal dan terkonfirmasi di leaderboard.](figures/fig3_progress.png)

## 4. Fitur: menerjemahkan fisika dan geografi ke dalam angka
Fitur adalah informasi yang kami berikan ke model. Kami menyusunnya dari pertanyaan sederhana: apa yang membuat satu kawasan memakai energi lebih banyak dari kawasan lain?

### 4.1 Ukuran rumah, bukan ukuran kawasan
Gas total kawasan terutama ditentukan oleh berapa banyak rumah di dalamnya, dan itu tidak menarik. Yang menarik adalah perilaku per rumah. Maka kami membuat rasio: luas lantai per meter (ukuran rata-rata rumah), luas per sertifikat energi, meter per sertifikat, dan sisa porsi jenis pemanas (100 dikurangi semua kategori pemanas yang tercatat, yang sebagian besar adalah pemanas listrik). Luas lantai per meter ternyata fitur terkuat untuk semua target konsumsi.

### 4.2 Menebak per rumah, bukan per kawasan
Model kami tidak menebak gas total secara langsung. Mereka menebak gas per meter, gas per meter persegi, gas per sertifikat, atau logaritmanya, lalu hasilnya dikalikan kembali. Bobot setiap baris diatur supaya kesalahan yang dihukum tetap kesalahan pada skala asli. Variasi cara menebak ini membuat model-model kami berbeda pandangan, dan perbedaan itulah yang membuat gabungannya kuat. Sebagian model juga memakai objektif Poisson, yang cocok untuk besaran berbentuk laju per rumah.

### 4.3 Belajar dari lingkungan sekitar
Kawasan yang berdekatan cenderung mirip: jaringan gasnya sama, iklimnya sama, umur bangunannya mirip. Kami menghitung rata-rata target di subzona, zona, petak iklim (gabungan tiga kolom iklim yang ternyata membentuk grid berisi sekitar 15 kawasan per petak), dan kawasan di sekitar nomor subzona (nomor subzona ternyata berurutan secara geografis). Supaya model tidak "mencontek", rata-rata ini selalu dihitung dari bagian data yang tidak sedang diuji.

### 4.4 Belajar dari kawasan yang paling mirip
Fitur yang paling berpengaruh di seluruh proyek ini menjawab pertanyaan: berapa konsumsi kawasan lain di sekitarnya yang stok bangunannya paling mirip? Untuk setiap kawasan kami mencari k kawasan terdekat dalam "ruang bangunan" (ukuran rumah, cakupan sertifikat, campuran era bangunan, porsi bangunan bersih) di dalam radius geografis tertentu, lalu memakai rata-rata konsumsi mereka sebagai fitur. Versi lanjutan memakai beberapa sudut pandang kemiripan sekaligus: era bangunan saja, bangunan ditambah sensus, dan ukuran saja. Fitur ini menurunkan galat EUI model tunggal sebesar 0,0022, perbaikan terbesar yang kami dapat dari satu ide.

### 4.5 Gambaran kawasan
Jumlah kawasan, total luas lantai, meter, dan sertifikat per subzona dan per zona, dihitung dari data latih dan uji sekaligus (tanpa target). Fitur ini memberi tahu model apakah sebuah kawasan bagian dari subzona besar atau kecil.

Gambar 3 merangkum seberapa besar tiap ide memperbaiki satu model, dan seberapa besar penggabungan serta koreksi (bagian 5) menambahnya.

![Gambar 3. Atas: perbaikan galat satu model dari setiap ide fitur dan cara pelatihan. Bawah: perbaikan dari menggabungkan model dan dari koreksi galat wilayah, diukur di ujian internal.](figures/fig5_gains.png)

## 5. Model dan cara menggabungkannya
### 5.1 Tiga jalur untuk lima target
Kelima target tidak diperlakukan sama (Gambar 4):
- Gas dan EUI dilayani satu keluarga model, karena keduanya besaran yang sama. Saat menggabungkan model, bobot untuk kolom gas dihitung dengan menekankan kawasan besar (yang paling menentukan galat gas), sedangkan bobot untuk kolom EUI tidak.
- Listrik punya keluarga model sendiri.
- Porsi bangunan bersih dihitung dengan rumus; adopsi disalin per subzona dan sisanya ditebak dengan model peluang.

![Gambar 4. Alur lengkap dari data mentah sampai file submisi.](figures/fig_arch.png)

### 5.2 Empat puluh satu model yang berbeda pandangan
Kami memakai beberapa jenis model sekaligus, karena tiap jenis melihat data dengan cara yang berbeda:
- **XGBoost dan LightGBM** (pohon keputusan bertingkat): kuat untuk data tabel dan cepat dilatih.
- **Jaringan saraf** (MLP), termasuk varian yang mengubah tiap angka menjadi pola gelombang (PLR) dan varian yang melatih 16 sub-jaringan sekaligus (TabM-mini), serta satu jaringan yang menebak gas dan listrik bersamaan.
- **Model "cari tetangga" (NCA):** jaringan yang belajar sendiri kawasan mana yang mirip, lalu menebak dari rata-rata kawasan mirip itu. Sendirian ia lebih lemah, tetapi kesalahannya paling berbeda dari model lain, sehingga berguna dalam gabungan.

**Tabel 2.** Komposisi gabungan model. Galat terbaik adalah galat model tunggal terbaik di keluarganya; bobot adalah porsi keluarga itu di gabungan akhir.

| Keluarga | Model gas/EUI | Model listrik | Galat terbaik EUI (gas) | Galat terbaik listrik | Bobot gas / EUI / listrik |
|---|---|---|---|---|---|
| XGBoost | 16 | 3 | 0,6564 (0,2486) | 0,3200 | 42% / 42% / 16% |
| LightGBM | 5 | 4 | 0,6566 (0,2488) | 0,3197 | 13% / 20% / 23% |
| MLP biasa | 2 | 1 | 0,6677 (0,2522) | 0,3213 | 5% / 1% / 3% |
| MLP PLR dan TabM | 3 | 2 | 0,6584 (0,2485) | 0,3188 | 27% / 19% / 25% |
| MLP gas + listrik | 1 | 2 | 0,6585 (0,2488) | 0,3186 | 6% / 11% / 26% |
| NCA | 1 | 1 | 0,6673 (0,2515) | 0,3217 | 8% / 7% / 8% |

Semua model dilatih dengan pembagian data yang sama dan seed yang terkunci, sehingga hasilnya bisa diulang persis. CatBoost juga kami coba dan hasilnya bagus, tetapi di GPU ia tidak menghasilkan angka yang sama persis setiap dijalankan, jadi kami keluarkan.

### 5.3 Menggabungkan dengan bobot yang menyesuaikan
Cara paling sederhana menggabungkan model adalah memberi tiap model bobot tetap. Kami menemukan bahwa model yang unggul di kawasan kecil tidak selalu unggul di kawasan besar. Karena itu bobot kami dibiarkan berubah perlahan mengikuti luas kawasan, jumlah meter, dan besar tebakannya sendiri. Bobot tetap dijaga tidak negatif, sehingga gabungan selalu berupa campuran yang masuk akal. Perubahan ini memperbaiki skor leaderboard 0,00009.

### 5.4 Koreksi galat wilayah dan penyesuaian akhir
Setelah digabung, galat model ternyata masih saling berkorelasi antar kawasan yang berdekatan dan mirip. Kalau tetangga-tetangga yang mirip sama-sama ditebak terlalu rendah, kemungkinan besar kawasan ini juga. Maka tebakan setiap kawasan digeser sedikit mengikuti rata-rata galat tetangga di zona, petak iklim, nomor subzona di sekitarnya, dan kawasan paling mirip. Besar geseran ditentukan di ujian internal, dengan tetangga yang hanya diambil dari bagian data yang tidak diuji.

Di akhir proyek kami melakukan audit menyeluruh: setiap temuan diulang oleh pemeriksa lain dengan kode sendiri sebelum dipakai. Dua masalah terbukti nyata dan tidak bisa diperbaiki oleh model mana pun:
- Kawasan dengan 2 sampai 3 meter ditebak sekitar 20% terlalu rendah, dan dengan 4 meter sekitar 3 sampai 7%, di ketiga target konsumsi. Hanya ada 877 kawasan seperti ini di data latih, terlalu sedikit untuk dipelajari model. Kami mengoreksinya dengan faktor pengali per kelompok.
- Tebakan gas di dekat batas satu juta kWh cenderung kebesaran, dan tujuh tebakan bahkan melewati batas yang tidak pernah dilewati data latih. Kami menariknya kembali dengan perhitungan distribusi normal terpotong dan memotong semua tebakan ke batas data latih.

Kedua penyesuaian ini memperbaiki skor leaderboard 0,00007, persis sebesar perkiraan ujian internal. Terakhir, 77 kawasan uji ternyata identik dengan kawasan latih (subzona, meter, sertifikat, luas, dan satu kolom sensus yang hampir unik semuanya sama), dan kelompok duplikat di data latih selalu punya target yang sama, sehingga targetnya kami salin.

## 6. Hasil
Gambar 2 menunjukkan perjalanan skor kami dari 0,2503 sampai 0,24913. Tidak ada satu lompatan besar; skor membaik dari banyak langkah kecil yang masing-masing terbukti di ujian internal.

Beberapa ide kami gagal dan kami catat semuanya di registry eksperimen. Contohnya: model lapis kedua yang belajar dari tebakan model lain, gabungan khusus untuk subzona baru, koreksi galat dengan model nonlinear, pelatihan dengan 20 bagian alih-alih 10, dan pencarian kebocoran data (urutan ID, nilai gas kembar, duplikat longgar). Kegagalan ini justru memberi keyakinan: setelah begitu banyak ide gagal memperbaiki ujian internal, kami tahu skor kami sudah dekat dengan batas yang bisa dicapai dari data ini.

## 7. Apa artinya bagi kebijakan
### 7.1 Model penjelas
Gabungan 41 model terlalu rumit untuk dibaca, dan sebagian fiturnya (rata-rata target wilayah) bukan sesuatu yang bisa diubah oleh pembuat kebijakan. Karena itu kami melatih satu model penjelas per target yang hanya memakai karakteristik kawasan yang bisa diukur: bangunan, era pembangunan, campuran pemanas, sosio-demografi, iklim, dan gambaran kawasan.

Untuk membaca model ini kami memakai SHAP. Bayangkan tebakan model sebagai tumpukan sumbangan: dimulai dari rata-rata nasional, lalu setiap fitur menambah atau mengurangi. SHAP menghitung besar sumbangan tiap fitur untuk setiap kawasan secara adil. Model penjelas ini cukup setia pada data: ia menjelaskan 75% variasi gas per meter, 56% variasi EUI, 49% variasi listrik per meter, dan 99% variasi adopsi. Sisanya sebagian besar adalah variasi antar rumah yang memang tidak bisa dijelaskan (bagian 2).

Nama kolom disamarkan panitia, jadi makna yang kami tulis adalah dugaan dari pola datanya, misalnya `hvac_dist_02` sebagai porsi rumah dengan pemanas gas jaringan.

### 7.2 Apa yang mendorong konsumsi
**Tabel 3.** Porsi sumbangan (SHAP) tiap kelompok fitur.

| Target | Sosio-demografi | Rasio fisik | Era bangunan | Dimensi bangunan | Iklim | Campuran pemanas |
|---|---|---|---|---|---|---|
| EUI (intensitas gas) | 30% | 25% | 11% | 11% | 9% | 3% |
| Gas per meter | 28% | 30% | 11% | 11% | 8% | 3% |
| Listrik per meter | 40% | 26% | 5% | 11% | 4% | 5% |
| Adopsi pemanas listrik | 15% | 0% | 0% | 0% | 1% | 81% |

![Gambar 5. Porsi sumbangan tiap kelompok fitur untuk tiap target.](figures/pol_families.png)

Empat pola paling penting untuk kebijakan (Gambar 6):

1. Ukuran rumah bekerja ke arah berlawanan untuk gas dan listrik. Rumah kecil (sekitar 30 m² per meter) memakai gas per meter persegi sekitar 65% lebih banyak daripada rumah besar (300 m²), karena dinding dan atapnya relatif luas dibanding ruang yang dipanaskan. Sebaliknya, rumah besar memakai listrik per rumah sekitar 75% lebih banyak. Artinya, program penghematan per meter persegi dan program pengurangan emisi total akan memilih kawasan yang berbeda.
2. Bangunan dari era tertentu lebih boros. Kawasan yang seluruhnya berisi bangunan dari periode `bld_era_15` punya EUI sekitar 7% lebih tinggi daripada kawasan tanpa bangunan periode itu, dan `bld_era_09` sekitar 4%. Inilah bagian konsumsi yang bisa dipangkas lewat perbaikan dinding, atap, dan jendela.
3. Iklim menggeser intensitas sekitar 15% antara kawasan beriklim paling ringan dan paling berat (`env_climate_05`). Dengan anggaran yang sama per rumah, perbaikan bangunan di kawasan beriklim berat menghemat lebih banyak.
4. Adopsi pemanas listrik ditentukan oleh infrastruktur, bukan oleh kemampuan warga. Di kawasan dengan porsi pemanas gas jaringan di bawah 40%, sumbangannya ke adopsi +4 sampai +12 poin persen. Di atas 80%, sumbangannya negatif. Faktor sosio-demografi hanya menyumbang 15%. Tanpa intervensi, elektrifikasi akan tetap terpusat di kawasan yang memang belum tersambung ke jaringan gas.

![Gambar 6. Hubungan beberapa fitur dengan sumbangannya (SHAP). Setiap titik adalah satu kawasan; garis merah adalah rata-ratanya. Atas: ukuran rumah terhadap EUI dan listrik, porsi gas jaringan terhadap adopsi. Bawah: iklim dan dua periode bangunan terhadap EUI.](figures/pol_dependence.png)

### 7.3 Peta trilema per zona
Untuk menjadikan temuan ini daftar kerja, kami menggabungkan nilai sebenarnya (kawasan latih) dan tebakan akhir (kawasan uji) untuk seluruh 613 ribu kawasan, lalu meringkasnya ke 330 zona dengan dua ukuran:

- Potensi hemat dari perbaikan bangunan: berapa gas yang hilang bila bagian intensitas yang disebabkan oleh stok bangunan (sumbangan era bangunan, dimensi bangunan, dan rasio fisik) ditekan ke rata-rata nasional. Totalnya 7,4 TWh, atau 5,2% dari 141,4 TWh konsumsi gas.
- Kenaikan beban listrik bila pemanas gas diganti pompa panas listrik: dengan asumsi 80% gas rumah tangga dipakai untuk pemanasan dan pompa panas menghasilkan 3 kWh panas dari 1 kWh listrik. Pada asumsi ini, konsumsi listrik zona naik rata-rata 95%, paling ringan 68% dan paling berat 121%.

![Gambar 7. Peta trilema: setiap titik adalah satu zona. Sumbu mendatar adalah intensitas gas, sumbu tegak adalah kenaikan beban listrik bila pemanasnya dielektrifikasi. Ukuran titik menunjukkan konsumsi gas, warna menunjukkan adopsi saat ini.](figures/pol_trilemma.png)

Garis median membagi zona menjadi empat kelompok (Tabel 4).

**Tabel 4.** Ringkasan empat kelompok zona.

| Kelompok | Zona | Porsi gas | Potensi hemat bangunan | Kenaikan beban rata-rata | Langkah yang cocok |
|---|---|---|---|---|---|
| Q1: boros dan jaringan rawan | 118 | 47% | 3,65 TWh | +102% | perbaiki bangunan dulu, lalu elektrifikasi bertahap |
| Q2: boros, jaringan masih longgar | 47 | 12% | 1,07 TWh | +90% | percepat pompa panas sekarang |
| Q3: hemat, jaringan rawan | 47 | 15% | 0,86 TWh | +99% | perkuat gardu dan tarif fleksibel dulu |
| Q4: hemat, jaringan longgar | 118 | 27% | 1,81 TWh | +87% | program standar |

### 7.4 Rekomendasi
1. Urutkan program per kelompok zona, jangan seragamkan. Q1 memuat hampir separuh konsumsi gas dan separuh potensi hemat bangunan. Di sini perbaikan bangunan harus datang lebih dulu daripada pompa panas. Kebutuhan panas yang turun lebih dulu memangkas tagihan dan emisi, sekaligus mengecilkan lonjakan beban saat elektrifikasi masuk. Zona REG_273, REG_270, dan REG_045 adalah kandidat pertama, dengan potensi hemat 6,7 sampai 8,6% dari konsumsi gas zona.
2. Jadikan Q2 kemenangan cepat untuk pompa panas. Intensitas gasnya tinggi tetapi jaringannya masih punya ruang. Subsidi pompa panas di sini memberi penurunan emisi terbesar tanpa menunggu penguatan gardu.
3. Ajak operator jaringan sejak awal di Q1 dan Q3. Zona seperti REG_290 (+121%), REG_259, dan REG_275 akan menanggung lebih dari dua kali lipat beban listrik bila pemanasnya dielektrifikasi penuh. Penguatan gardu, tarif berdasarkan waktu pemakaian, dan pompa panas dengan penyimpan panas perlu direncanakan sebelum subsidi besar dibuka.
4. Beri insentif khusus untuk kawasan yang sudah tersambung gas. Karena adopsi saat ini hampir hanya terjadi di kawasan yang kurang tersambung ke jaringan gas, kawasan dengan porsi gas jaringan di atas 80% (yang menampung sebagian besar rumah tangga) butuh dorongan tambahan, misalnya bantuan biaya pemasangan dan pembongkaran boiler lama.
5. Pisahkan target efisiensi dan target emisi. Rumah kecil boros per meter persegi, rumah besar boros per rumah. Program yang mengejar keterjangkauan sebaiknya menyasar kawasan rumah kecil dan bangunan lama, sedangkan program emisi menyasar zona dengan volume gas terbesar.

Batasannya: SHAP menunjukkan hubungan dalam data, bukan sebab-akibat yang pasti. Skenario beban listrik adalah perhitungan fisika sederhana di atas tebakan model, dengan asumsi yang kami tulis terbuka; urutan antar zona lebih bisa dipegang daripada angka mutlaknya. Indeks per zona dihitung dari tebakan yang sama dengan file submisi, sehingga langsung bisa dipakai sebagai daftar kerja.

## 8. Kode dan reproduksibilitas
Seluruh solusi ada di satu notebook yang menulis semua modul kodenya sendiri, lalu menjalankan pelatihan, penggabungan, penyesuaian akhir, evaluasi, dan analisis kebijakan secara berurutan. Seed dikunci di semua tahap, tidak ada batas waktu pelatihan, dan versi pustaka dicatat. Kami menjalankan ulang seluruh pelatihan dari awal: setiap model menghasilkan angka yang identik sampai byte terakhir dengan versi pengembangan, dan file submisi yang dihasilkan sama persis dengan file yang kami pilih (sidik sha256 aec745ab...d66e). Notebook kedua bisa membuat ulang submisi yang sama dari model yang sudah tersimpan, untuk pemeriksa yang tidak punya waktu melatih ulang sekitar 20 jam.

## 9. Penutup
Pelajaran terbesar dari kompetisi ini sederhana: waktu yang dipakai untuk memahami data jauh lebih berharga daripada waktu yang dipakai untuk mencari model yang lebih besar. Dua rumus fisika, satu sifat sensus, dan satu pemeriksaan cara pembagian data menyelesaikan dua target dan menentukan arah seluruh pekerjaan. Sisanya adalah disiplin: setiap ide diuji dengan jujur sebelum dipakai, dan setiap angka bisa diulang. Di sisi kebijakan, model yang sama bisa diubah menjadi peta prioritas yang menjawab pertanyaan praktis: zona mana yang perlu dibenahi bangunannya dulu, mana yang siap beralih ke listrik, dan mana yang jaringannya harus diperkuat lebih dulu.
