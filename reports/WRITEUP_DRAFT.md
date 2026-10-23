# Analisis Struktur Data sebagai Dasar Pemodelan: Prediksi Trilema Energi Perkotaan dan Prioritas Kebijakan per Zona

DICE FEST 2026, Machine Learning. Tim BISMILLAH

## Ringkasan
Kompetisi ini menuntut prediksi lima besaran energi untuk setiap kawasan permukiman, yaitu konsumsi gas, konsumsi listrik, intensitas gas per meter persegi, porsi bangunan bersih, dan porsi rumah tangga yang telah memakai pemanas listrik. Data latih memuat 429 ribu kawasan, sedangkan data uji memuat 184 ribu kawasan.

Sebelum pemodelan, kami melakukan analisis eksploratif terhadap struktur data. Analisis tersebut menunjukkan bahwa satu dari lima target dapat dihitung secara eksak dari kolom yang tersedia, satu target lain sudah diketahui nilainya untuk 92% kawasan uji, dan dua target lainnya terhubung oleh identitas fisis sederhana. Dengan demikian, upaya pemodelan dapat dipusatkan pada target yang tidak dapat diturunkan secara langsung.

Target tersebut dimodelkan melalui tiga komponen: fitur yang merepresentasikan mekanisme konsumsi energi (ukuran rumah, umur bangunan, dan kemiripan dengan kawasan sekitar), gabungan (*ensemble*) 41 model dengan pola galat yang berbeda, serta skema validasi yang meniru pemisahan data uji sehingga perbaikan pada validasi konsisten dengan perbaikan pada *leaderboard*. Skor akhir pada *leaderboard* publik adalah 0,24913.

Untuk analisis kebijakan, kami melatih model penjelas yang hanya memakai karakteristik kawasan yang terukur, lalu menyusun peta prioritas untuk 330 zona. Peta ini membedakan zona yang memerlukan perbaikan bangunan lebih dahulu, zona yang siap beralih ke pompa panas listrik (*heat pump*), dan zona yang jaringan listriknya perlu diperkuat sebelum elektrifikasi.

| Hasil utama | |
|---|---|
| Skor *leaderboard* publik | 0,24913 dari gabungan 41 model; *notebook* menghasilkan ulang berkas submisi yang identik |
| Struktur data | 1 target dihitung eksak dengan rumus, 1 target sudah diketahui untuk 92% kawasan uji, gas dan EUI terhubung melalui luas lantai |
| Pendorong konsumsi | ukuran rumah (berpengaruh berlawanan arah terhadap gas dan listrik), stok bangunan lama, dan iklim; 81% kontribusi pada adopsi pemanas listrik berasal dari infrastruktur pemanas yang ada |
| Peta trilema | 118 zona (47% konsumsi gas) memerlukan perbaikan bangunan sebelum elektrifikasi; potensi penghematan 7,4 TWh; elektrifikasi pemanas penuh menaikkan beban listrik zona rata-rata 95% |

## 1. Latar Belakang
Kota yang hendak beralih dari pemanas berbahan bakar gas menghadapi tiga tuntutan yang saling bertentangan. Emisi harus turun, jaringan listrik harus tetap andal ketika pemanas beralih ke listrik, dan biaya energi rumah tangga harus tetap terjangkau. Kelima target kompetisi mencerminkan ketiga aspek tersebut.

| Target | Makna | Aspek trilema |
|---|---|---|
| gas total (kWh) | konsumsi gas tahunan kawasan | emisi |
| listrik total (kWh) | konsumsi listrik tahunan kawasan | beban jaringan |
| EUI (kWh/m²) | konsumsi gas per meter persegi lantai (*energy use intensity*), ukuran efisiensi bangunan | emisi dan biaya |
| porsi bangunan bersih (%) | kualitas stok bangunan | emisi |
| adopsi pemanas listrik (%) | tingkat kemajuan transisi | transisi |

Setiap baris data mewakili satu kawasan kecil (kurang lebih setara dengan satu wilayah kode pos) yang berisi rata-rata dua puluh rumah tangga. Data memuat 201 kolom fitur yang namanya disamarkan oleh panitia, meliputi dimensi bangunan, periode pembangunan, komposisi jenis pemanas, iklim, dan sosiodemografi, serta dua kode lokasi (330 zona dan sekitar 150 ribu subzona).

Metrik evaluasi adalah rata-rata galat kelima target. Galat setiap target berupa RMSE yang dinormalisasi oleh simpangan baku target tersebut (RMSE/σ). Prediksi konstan berupa nilai rata-rata menghasilkan skor sekitar 1,0. Karena setiap target dinilai secara terpisah, setiap target dapat ditangani dengan pendekatan tersendiri. Pada model akhir, EUI menyumbang sekitar separuh total galat, listrik seperempat, gas seperlima, dan adopsi sisanya. Alokasi upaya pemodelan mengikuti proporsi tersebut.

## 2. Analisis Data Eksploratif
### 2.1 Dua Identitas Fisis Tersembunyi
Hasil kali EUI dan kolom `bld_dim_06` selalu sama dengan gas total hingga presisi mesin (selisih terbesar 7 × 10⁻¹⁶). Hal ini menunjukkan bahwa `bld_dim_06` adalah luas lantai dan bahwa gas total serta EUI merupakan satu besaran yang dinyatakan dalam dua bentuk, yaitu nilai total dan nilai per satuan luas lantai. Porsi bangunan bersih juga selalu sama dengan 100 × `bld_dim_02` / `bld_dim_01` yang dibulatkan hingga dua desimal pada seluruh 429 ribu baris. Dengan demikian, target ini dihitung langsung dan tidak perlu diprediksi.

### 2.2 Keseragaman Adopsi dalam Subzona
Nilai adopsi pemanas listrik selalu seragam di dalam satu subzona (subzona merupakan unit sensus). Karena 92% kawasan uji berada di subzona yang juga terdapat pada data latih, nilai adopsi kawasan tersebut dapat diisi langsung dari nilai subzonanya. Untuk 8% sisanya, kami memanfaatkan sifat kolom komposisi jenis pemanas yang merupakan pecahan dari jumlah rumah tangga. Bilangan bulat terkecil yang membuat semua pecahan tersebut menjadi bilangan bulat adalah jumlah rumah tangga di subzona (K). Atas dasar itu, adopsi dimodelkan secara probabilistik sebagai banyaknya rumah tangga pengguna pemanas listrik dari K rumah tangga.

### 2.3 Galat yang Tidak Dapat Direduksi
Konsumsi energi satu rumah tangga dapat berbeda jauh dari tetangganya karena faktor yang tidak tercatat dalam data, seperti jumlah penghuni, kebiasaan, dan pengaturan termostat. Ketika konsumsi beberapa rumah tangga dijumlahkan, sebagian variasi tersebut saling meniadakan, dan sisanya makin kecil seiring dengan bertambahnya jumlah meter di kawasan. Pola ini tampak jelas pada data (Gambar 1): sebaran galat mengikuti a + b/M, dengan M sebagai jumlah meter. Sekitar tiga perempat galat gas dan listrik berasal dari variasi antarrumah tangga ini. Oleh karena itu, peningkatan kapasitas model hanya memberikan perbaikan kecil. Perbaikan yang lebih berarti diperoleh dengan mengurangi galat acak model itu sendiri.

![Gambar 1. Sebaran galat model per kelompok jumlah meter. Titik-titik mengikuti kurva a + b/M: makin banyak rumah tangga dalam kawasan, makin kecil galatnya. Garis putus-putus menunjukkan komponen galat yang masih dapat direduksi oleh model.](figures/fig2_noise_floor.png)

### 2.4 Batas Nilai pada Data
Nilai gas pada data latih tidak pernah melebihi 999.765 kWh sehingga tampak terpotong tepat di bawah satu juta. Kawasan dengan jumlah meter sangat sedikit (2 hingga 4) juga menunjukkan perilaku yang berbeda dari kawasan lain. Kedua hal ini ditangani pada tahap akhir (subbab 5.4).

## 3. Skema Validasi
Kinerja model pada data latih tidak menjamin kinerja yang sama pada data uji sehingga skema validasi perlu meniru cara panitia memisahkan data. Kami memeriksa empat indikator. Pertama, data uji berjumlah tepat 30% dari total. Kedua, kawasan uji tersebar merata di semua zona. Ketiga, porsi kawasan uji yang berada di subzona baru sebesar 8,14%, sesuai dengan nilai harapan pengacakan sebesar 8,17%. Keempat, model klasifikasi yang dilatih untuk membedakan baris latih dari baris uji (*adversarial validation*) tidak lebih baik daripada klasifikasi acak. Kesimpulannya, data uji diambil secara acak per baris.

Berdasarkan temuan tersebut, skema yang dipakai adalah validasi silang acak sepuluh lipatan (*10-fold cross-validation*): data latih dibagi secara acak menjadi sepuluh bagian, lalu model dilatih pada sembilan bagian dan diuji pada satu bagian secara bergiliran. Pemilihan skema ini berpengaruh besar. Jika data dibagi per zona, seperti yang sering dianjurkan untuk data spasial, fitur lokasi tampak menurunkan kinerja model. Padahal, pada data uji yang sebenarnya, fitur yang sama justru termasuk fitur yang paling bermanfaat (Tabel 1).

**Tabel 1.** Galat LightGBM (makin kecil makin baik) dengan dan tanpa fitur rata-rata target wilayah pada dua skema pembagian data.

| Skema pembagian data | Fitur rata-rata target wilayah | Gas | EUI | Listrik |
|---|---|---|---|---|
| acak per baris (sesuai data uji) | tanpa | 0,2573 | 0,6721 | 0,3262 |
| acak per baris (sesuai data uji) | dengan | 0,2562 | 0,6699 | 0,3255 |
| per zona | tanpa | 0,2708 | 0,6923 | 0,3302 |
| per zona | dengan | 0,3637 | 0,8331 | 0,3474 |

Untuk tahap setelah penggabungan model (koreksi dan penyesuaian akhir), 30% data latih disisihkan sebagai set *holdout* internal. Semua model dilatih ulang tanpa bagian ini, kemudian setiap usulan perbaikan dievaluasi padanya. Set *holdout* ini terbukti menjadi penduga yang akurat: pada setiap langkah, besar perbaikan skor yang diperkirakannya hampir sama dengan besar perbaikan pada *leaderboard* (Gambar 2). Oleh karena itu, tidak ada perubahan yang dipilih berdasarkan *leaderboard* publik, yang hanya memakai 30% data uji.

![Gambar 2. Perkembangan skor leaderboard publik. Setiap langkah dipilih berdasarkan set holdout internal dan terkonfirmasi pada leaderboard.](figures/fig3_progress.png)

## 4. Rekayasa Fitur
Fitur disusun berdasarkan satu pertanyaan: faktor apa yang membuat suatu kawasan mengonsumsi energi lebih banyak daripada kawasan lain?

### 4.1 Ukuran Rumah, Bukan Ukuran Kawasan
Gas total suatu kawasan terutama ditentukan oleh jumlah rumah tangga di dalamnya. Informasi yang lebih bermakna bagi model adalah konsumsi per rumah tangga. Oleh karena itu, kami membentuk fitur rasio, yaitu luas lantai per meter (ukuran rata-rata rumah), luas per sertifikat energi, meter per sertifikat, dan sisa porsi jenis pemanas (100 dikurangi jumlah semua kategori pemanas yang tercatat, yang sebagian besar merupakan pemanas listrik). Luas lantai per meter merupakan fitur dengan kontribusi terbesar untuk semua target konsumsi.

### 4.2 Target per Rumah Tangga, Bukan per Kawasan
Model tidak memprediksi gas total secara langsung. Target yang diprediksi adalah gas per meter, gas per meter persegi, gas per sertifikat, atau logaritmanya, kemudian hasilnya dikalikan kembali ke skala kawasan. Bobot setiap baris diatur agar fungsi kerugian tetap setara dengan galat pada skala asli. Variasi transformasi target ini menghasilkan model dengan pola galat yang berbeda, dan keragaman inilah yang membuat gabungannya lebih akurat. Sebagian model juga memakai fungsi objektif Poisson yang sesuai untuk besaran berbentuk laju per rumah tangga.

### 4.3 Informasi dari Lingkungan Sekitar
Kawasan yang berdekatan cenderung serupa karena berbagi jaringan gas dan iklim yang sama serta memiliki umur bangunan yang mirip. Kami menghitung rata-rata target pada tingkat subzona, zona, petak iklim (kombinasi tiga kolom iklim yang ternyata membentuk kisi dengan sekitar lima belas kawasan per petak), dan kawasan dengan nomor subzona berdekatan (nomor subzona ternyata berurutan secara geografis). Untuk mencegah kebocoran informasi target, rata-rata ini selalu dihitung secara *out-of-fold*, yaitu dari bagian data yang tidak sedang diuji.

### 4.4 Informasi dari Kawasan Termirip
Fitur dengan pengaruh terbesar dalam pekerjaan ini menjawab pertanyaan berikut: berapa konsumsi kawasan sekitar yang stok bangunannya paling mirip? Untuk setiap kawasan, dicari k kawasan terdekat (*k-nearest neighbors*) dalam ruang karakteristik bangunan, yaitu ukuran rumah, cakupan sertifikat, komposisi periode bangunan, dan porsi bangunan bersih, di dalam radius geografis tertentu. Rata-rata konsumsi kawasan-kawasan tersebut kemudian dipakai sebagai fitur. Versi lanjutannya memakai beberapa definisi kemiripan sekaligus, yaitu periode bangunan saja, bangunan ditambah variabel sensus, dan ukuran saja. Fitur ini menurunkan galat EUI model tunggal sebesar 0,0022, perbaikan terbesar yang diperoleh dari satu gagasan.

### 4.5 Agregat Kawasan
Jumlah kawasan, total luas lantai, jumlah meter, dan jumlah sertifikat per subzona dan per zona dihitung dari gabungan data latih dan data uji tanpa memakai target. Fitur ini memberi model informasi mengenai ukuran subzona tempat suatu kawasan berada.

Gambar 3 merangkum besar perbaikan yang diberikan setiap gagasan pada model tunggal, serta tambahan perbaikan dari penggabungan model dan koreksi galat (bagian 5).

![Gambar 3. Atas: penurunan galat model tunggal dari setiap gagasan fitur dan skema pelatihan. Bawah: penurunan galat dari penggabungan model dan koreksi galat wilayah, diukur pada set holdout internal.](figures/fig5_gains.png)

## 5. Pemodelan dan Penggabungan Model
### 5.1 Tiga Jalur untuk Lima Target
Kelima target ditangani dengan pendekatan yang berbeda (Gambar 4).
- Gas dan EUI ditangani oleh satu kelompok model karena keduanya merupakan besaran yang sama. Pada tahap penggabungan, bobot untuk kolom gas dihitung dengan penekanan pada kawasan besar yang paling menentukan galat gas, sedangkan bobot untuk kolom EUI dihitung tanpa penekanan tersebut.
- Listrik ditangani oleh kelompok model tersendiri.
- Porsi bangunan bersih dihitung dengan rumus. Adopsi diisi dari nilai subzona, sedangkan kawasan di subzona baru (8%) diprediksi dengan model probabilistik.

![Gambar 4. Alur lengkap dari data mentah hingga berkas submisi.](figures/fig_arch.png)

### 5.2 Empat Puluh Satu Model dengan Pola Galat Berbeda
Kami memakai beberapa jenis model karena setiap jenis menangkap pola data dengan cara yang berbeda.
- **XGBoost dan LightGBM** (*gradient boosted decision trees*): andal untuk data tabular dan cepat dilatih.
- **Jaringan saraf (MLP)**: mencakup varian yang mentransformasi setiap fitur numerik menjadi representasi periodik (PLR), varian yang melatih enam belas subjaringan secara bersamaan (TabM-mini), dan satu jaringan yang memprediksi gas dan listrik secara bersamaan.
- **Model berbasis tetangga (NCA)**: jaringan yang mempelajari ukuran kemiripan antarkawasan, lalu memprediksi berdasarkan rata-rata kawasan yang mirip. Sebagai model tunggal, kinerjanya lebih rendah, tetapi pola galatnya paling berbeda dari model lain sehingga bermanfaat dalam gabungan.

**Tabel 2.** Komposisi gabungan model. Galat terbaik adalah galat model tunggal terbaik dalam kelompoknya; bobot adalah porsi kelompok tersebut dalam gabungan akhir.

| Kelompok | Model gas/EUI | Model listrik | Galat terbaik EUI (gas) | Galat terbaik listrik | Bobot gas / EUI / listrik |
|---|---|---|---|---|---|
| XGBoost | 16 | 3 | 0,6564 (0,2486) | 0,3200 | 42% / 42% / 16% |
| LightGBM | 5 | 4 | 0,6566 (0,2488) | 0,3197 | 13% / 20% / 23% |
| MLP standar | 2 | 1 | 0,6677 (0,2522) | 0,3213 | 5% / 1% / 3% |
| MLP PLR dan TabM | 3 | 2 | 0,6584 (0,2485) | 0,3188 | 27% / 19% / 25% |
| MLP gas + listrik | 1 | 2 | 0,6585 (0,2488) | 0,3186 | 6% / 11% / 26% |
| NCA | 1 | 1 | 0,6673 (0,2515) | 0,3217 | 8% / 7% / 8% |

Semua model dilatih dengan pembagian data yang sama dan *seed* yang dikunci sehingga hasilnya dapat direproduksi secara identik. CatBoost juga diuji dan memberikan hasil yang baik, tetapi pada GPU model ini tidak menghasilkan keluaran yang identik pada setiap eksekusi sehingga tidak disertakan.

### 5.3 Penggabungan dengan Bobot Adaptif
Cara penggabungan yang paling sederhana adalah memberi setiap model bobot tetap. Namun, kami menemukan bahwa model yang unggul pada kawasan kecil tidak selalu unggul pada kawasan besar. Oleh karena itu, bobot dibuat berubah secara halus sebagai fungsi luas kawasan, jumlah meter, dan besar prediksi itu sendiri. Bobot dibatasi agar tidak negatif sehingga hasil gabungan tetap merupakan campuran prediksi yang dapat ditafsirkan. Perubahan ini memperbaiki skor *leaderboard* sebesar 0,00009.

### 5.4 Koreksi Galat Wilayah dan Penyesuaian Akhir
Setelah penggabungan, galat model masih berkorelasi antarkawasan yang berdekatan dan serupa. Jika kawasan tetangga yang serupa sama-sama diprediksi terlalu rendah, kawasan yang bersangkutan kemungkinan besar juga diprediksi terlalu rendah. Oleh karena itu, prediksi setiap kawasan dikoreksi sebesar sebagian dari rata-rata galat tetangganya pada tingkat zona, petak iklim, nomor subzona yang berdekatan, dan kawasan termirip. Besar koreksi ditentukan pada set *holdout* internal, dengan tetangga yang hanya diambil dari bagian data yang tidak sedang diuji.

Pada akhir proyek, kami melakukan audit menyeluruh. Setiap temuan direplikasi secara independen oleh pemeriksa lain dengan kode tersendiri sebelum dipakai. Dua masalah terbukti nyata dan tidak dapat diatasi oleh model mana pun.
- Kawasan dengan 2 hingga 3 meter diprediksi sekitar 20% terlalu rendah, sedangkan kawasan dengan 4 meter diprediksi sekitar 3% hingga 7% terlalu rendah pada ketiga target konsumsi. Data latih hanya memuat 877 kawasan seperti ini, jumlah yang terlalu sedikit untuk dipelajari model. Masalah ini dikoreksi dengan faktor pengali per kelompok.
- Prediksi gas yang mendekati batas satu juta kWh cenderung terlalu tinggi (*overestimate*), dan tujuh prediksi bahkan melampaui nilai maksimum data latih. Prediksi di sekitar batas tersebut dikoreksi dengan pendekatan distribusi normal terpancung, lalu semua prediksi dipotong pada nilai maksimum data latih.

Kedua penyesuaian ini memperbaiki skor *leaderboard* sebesar 0,00007, sama dengan perkiraan set *holdout* internal. Terakhir, 77 kawasan uji ternyata identik dengan kawasan latih (subzona, jumlah meter, jumlah sertifikat, luas, dan satu kolom sensus yang hampir unik semuanya sama). Kelompok duplikat pada data latih selalu memiliki nilai target yang sama sehingga target kawasan uji tersebut disalin dari pasangannya.

## 6. Hasil
Gambar 2 menunjukkan perkembangan skor dari 0,2503 hingga 0,24913. Perbaikan skor tidak berasal dari satu lompatan besar, tetapi dari akumulasi banyak perbaikan kecil yang masing-masing telah diverifikasi pada set *holdout* internal.

Sejumlah pendekatan tidak memberikan perbaikan, dan seluruhnya tercatat dalam register eksperimen. Pendekatan tersebut antara lain model lapis kedua (*stacking*) yang dilatih dari prediksi model lain, gabungan khusus untuk subzona baru, koreksi galat dengan model nonlinear, validasi silang dua puluh lipatan sebagai pengganti sepuluh lipatan, dan penelusuran kebocoran data (urutan ID, nilai gas kembar, dan duplikat longgar). Banyaknya pendekatan yang tidak memperbaiki skor pada set *holdout* internal mengindikasikan bahwa skor akhir sudah mendekati batas yang dapat dicapai dari data ini.

## 7. Implikasi Kebijakan
### 7.1 Model Penjelas
Gabungan 41 model terlalu kompleks untuk ditafsirkan secara langsung, dan sebagian fiturnya (rata-rata target wilayah) bukan variabel yang dapat diintervensi oleh pembuat kebijakan. Oleh karena itu, kami melatih satu model penjelas untuk setiap target dengan hanya memakai karakteristik kawasan yang terukur, yaitu bangunan, periode pembangunan, komposisi pemanas, sosiodemografi, iklim, dan agregat kawasan.

Model penjelas ditafsirkan dengan SHAP (*SHapley Additive exPlanations*). SHAP menguraikan setiap prediksi menjadi nilai dasar, yaitu rata-rata prediksi pada seluruh data, ditambah kontribusi aditif dari setiap fitur. Kontribusi tersebut dihitung berdasarkan nilai Shapley dari teori permainan kooperatif. Model penjelas ini memiliki kecocokan yang memadai dengan data: model menjelaskan 75% variasi gas per meter, 56% variasi EUI, 49% variasi listrik per meter, dan 99% variasi adopsi. Sebagian besar variasi yang tersisa adalah variasi antarrumah tangga yang memang tidak dapat dijelaskan (subbab 2.3).

Karena nama kolom disamarkan oleh panitia, makna kolom yang kami tuliskan merupakan dugaan berdasarkan pola data, misalnya `hvac_dist_02` sebagai porsi rumah tangga dengan pemanas gas jaringan.

### 7.2 Faktor Pendorong Konsumsi
**Tabel 3.** Porsi kontribusi (SHAP) setiap kelompok fitur.

| Target | Sosiodemografi | Rasio fisik | Periode bangunan | Dimensi bangunan | Iklim | Komposisi pemanas |
|---|---|---|---|---|---|---|
| EUI (intensitas gas) | 30% | 25% | 11% | 11% | 9% | 3% |
| Gas per meter | 28% | 30% | 11% | 11% | 8% | 3% |
| Listrik per meter | 40% | 26% | 5% | 11% | 4% | 5% |
| Adopsi pemanas listrik | 15% | 0% | 0% | 0% | 1% | 81% |

![Gambar 5. Porsi kontribusi setiap kelompok fitur untuk setiap target.](figures/pol_families.png)

Empat pola yang paling relevan bagi kebijakan adalah sebagai berikut (Gambar 6).

1. Ukuran rumah berpengaruh berlawanan arah terhadap gas dan listrik. Rumah kecil (sekitar 30 m² per meter) mengonsumsi gas per meter persegi sekitar 65% lebih banyak daripada rumah besar (300 m²) karena luas dinding dan atapnya relatif besar terhadap ruang yang dipanaskan. Sebaliknya, rumah besar mengonsumsi listrik per rumah sekitar 75% lebih banyak. Artinya, program efisiensi per meter persegi dan program pengurangan emisi total akan menyasar kawasan yang berbeda.
2. Bangunan dari periode tertentu memiliki intensitas energi yang lebih tinggi. Kawasan yang seluruh bangunannya berasal dari periode `bld_era_15` memiliki EUI sekitar 7% lebih tinggi daripada kawasan tanpa bangunan dari periode tersebut, sedangkan untuk `bld_era_09` selisihnya sekitar 4%. Bagian konsumsi inilah yang dapat ditekan melalui perbaikan dinding, atap, dan jendela.
3. Iklim menggeser intensitas energi sekitar 15% antara kawasan dengan iklim paling ringan dan paling berat (`env_climate_05`). Dengan anggaran per rumah yang sama, perbaikan bangunan di kawasan beriklim berat menghasilkan penghematan yang lebih besar.
4. Adopsi pemanas listrik lebih ditentukan oleh infrastruktur daripada oleh karakteristik sosiodemografi warga. Pada kawasan dengan porsi pemanas gas jaringan di bawah 40%, kontribusi faktor ini terhadap adopsi bernilai positif, yaitu +4 hingga +12 poin persentase. Pada porsi di atas 80%, kontribusinya negatif. Faktor sosiodemografi hanya menyumbang 15%. Tanpa intervensi, elektrifikasi akan tetap terkonsentrasi di kawasan yang belum tersambung ke jaringan gas.

![Gambar 6. Hubungan beberapa fitur dengan kontribusinya (SHAP). Setiap titik adalah satu kawasan dan garis merah menunjukkan rata-ratanya. Atas: ukuran rumah terhadap EUI dan listrik, serta porsi gas jaringan terhadap adopsi. Bawah: iklim dan dua periode bangunan terhadap EUI.](figures/pol_dependence.png)

### 7.3 Peta Trilema per Zona
Untuk menerjemahkan temuan ini menjadi prioritas program, nilai aktual (kawasan latih) dan prediksi akhir (kawasan uji) untuk seluruh 613 ribu kawasan digabungkan, lalu diringkas ke tingkat 330 zona dengan dua ukuran berikut.

- Potensi penghematan dari perbaikan bangunan, yaitu besar konsumsi gas yang berkurang apabila komponen intensitas yang disebabkan oleh stok bangunan (kontribusi periode bangunan, dimensi bangunan, dan rasio fisik) diturunkan ke rata-rata nasional. Totalnya 7,4 TWh, atau 5,2% dari 141,4 TWh konsumsi gas.
- Kenaikan beban listrik apabila pemanas gas diganti dengan pompa panas listrik, dengan asumsi 80% konsumsi gas rumah tangga dipakai untuk pemanasan dan pompa panas menghasilkan 3 kWh panas dari 1 kWh listrik. Dengan asumsi ini, konsumsi listrik zona naik rata-rata 95%, dengan kenaikan terendah 68% dan tertinggi 121%.

![Gambar 7. Peta trilema: setiap titik mewakili satu zona. Sumbu horizontal menunjukkan intensitas gas dan sumbu vertikal menunjukkan kenaikan beban listrik apabila pemanas dielektrifikasi. Ukuran titik menunjukkan konsumsi gas dan warna menunjukkan tingkat adopsi saat ini.](figures/pol_trilemma.png)

Garis median membagi zona menjadi empat kuadran (Tabel 4).

**Tabel 4.** Ringkasan empat kuadran zona.

| Kuadran | Jumlah zona | Porsi gas | Potensi penghematan bangunan | Rata-rata kenaikan beban | Langkah yang sesuai |
|---|---|---|---|---|---|
| Q1: intensitas tinggi, kenaikan beban tinggi | 118 | 47% | 3,65 TWh | +102% | perbaikan bangunan lebih dahulu, kemudian elektrifikasi bertahap |
| Q2: intensitas tinggi, kenaikan beban rendah | 47 | 12% | 1,07 TWh | +90% | percepatan pemasangan pompa panas |
| Q3: intensitas rendah, kenaikan beban tinggi | 47 | 15% | 0,86 TWh | +99% | penguatan gardu dan tarif fleksibel lebih dahulu |
| Q4: intensitas rendah, kenaikan beban rendah | 118 | 27% | 1,81 TWh | +87% | program standar |

### 7.4 Rekomendasi
1. Program perlu diurutkan per kuadran, tidak diseragamkan. Q1 memuat hampir separuh konsumsi gas dan separuh potensi penghematan bangunan. Pada kuadran ini, perbaikan bangunan perlu didahulukan daripada pemasangan pompa panas. Penurunan kebutuhan panas yang lebih awal menurunkan tagihan dan emisi, sekaligus memperkecil lonjakan beban ketika elektrifikasi dimulai. Zona REG_273, REG_270, dan REG_045 merupakan kandidat pertama dengan potensi penghematan 6,7% hingga 8,6% dari konsumsi gas zona.
2. Q2 menjadi prioritas jangka pendek untuk pompa panas. Intensitas gasnya tinggi, tetapi jaringan listriknya masih memiliki kapasitas cadangan. Subsidi pompa panas pada kuadran ini berpotensi memberikan penurunan emisi terbesar tanpa harus menunggu penguatan gardu.
3. Operator jaringan perlu dilibatkan sejak awal pada zona Q1 dan Q3. Zona seperti REG_290 (+121%), REG_259, dan REG_275 akan menanggung beban listrik lebih dari dua kali lipat apabila pemanasnya dielektrifikasi penuh. Penguatan gardu, tarif berbasis waktu pemakaian, dan pompa panas dengan penyimpan panas perlu direncanakan sebelum subsidi berskala besar dibuka.
4. Insentif khusus diperlukan untuk kawasan yang sudah tersambung ke jaringan gas. Karena adopsi saat ini hampir hanya terjadi di kawasan yang kurang tersambung ke jaringan gas, kawasan dengan porsi gas jaringan di atas 80% (yang menampung sebagian besar rumah tangga) memerlukan dorongan tambahan, misalnya bantuan biaya pemasangan dan pembongkaran ketel pemanas (*boiler*) lama.
5. Target efisiensi dan target emisi perlu dipisahkan. Rumah kecil memiliki intensitas gas per meter persegi yang tinggi, sedangkan rumah besar memiliki konsumsi per rumah yang tinggi. Program yang berfokus pada keterjangkauan sebaiknya menyasar kawasan dengan rumah kecil dan bangunan lama, sedangkan program pengurangan emisi menyasar zona dengan volume konsumsi gas terbesar.

Analisis ini memiliki beberapa keterbatasan. SHAP menunjukkan asosiasi dalam data dan belum membuktikan hubungan sebab-akibat. Skenario beban listrik merupakan perhitungan fisis sederhana berdasarkan prediksi model dengan asumsi yang dinyatakan secara eksplisit. Peringkat antarzona lebih dapat diandalkan daripada nilai absolutnya. Indeks per zona dihitung dari prediksi yang sama dengan berkas submisi sehingga dapat langsung dipakai sebagai daftar prioritas.

## 8. Kode dan Keterulangan Hasil
Seluruh solusi terdapat dalam satu *notebook* yang menuliskan semua modul kodenya sendiri, lalu menjalankan pelatihan, penggabungan, penyesuaian akhir, evaluasi, dan analisis kebijakan secara berurutan. *Seed* dikunci pada semua tahap, tidak ada batas waktu pelatihan, dan versi pustaka dicatat. Seluruh pelatihan telah dijalankan ulang dari awal. Setiap model menghasilkan keluaran yang identik hingga tingkat *byte* dengan versi pengembangan, dan berkas submisi yang dihasilkan identik dengan berkas yang dipilih (nilai *hash* SHA-256 aec745ab...d66e). *Notebook* kedua dapat menghasilkan ulang submisi yang sama dari model yang telah disimpan. *Notebook* ini disediakan bagi pemeriksa yang tidak memiliki waktu untuk melatih ulang seluruh model selama sekitar 20 jam.

## 9. Simpulan
Pengalaman dalam kompetisi ini menunjukkan bahwa waktu yang dicurahkan untuk memahami struktur data jauh lebih bernilai daripada waktu yang dipakai untuk mencari model yang lebih besar. Dua identitas fisis, satu sifat data sensus, dan satu pemeriksaan skema pemisahan data cukup untuk menyelesaikan dua target dan menentukan arah seluruh pekerjaan. Pekerjaan selanjutnya bertumpu pada prosedur yang konsisten: setiap gagasan divalidasi sebelum dipakai dan setiap hasil dapat direproduksi. Dari sisi kebijakan, model yang sama dapat diterjemahkan menjadi peta prioritas yang menjawab pertanyaan praktis, yaitu zona mana yang memerlukan perbaikan bangunan lebih dahulu, zona mana yang siap beralih ke pemanas listrik, dan zona mana yang jaringan listriknya perlu diperkuat terlebih dahulu.
