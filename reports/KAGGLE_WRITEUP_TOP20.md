# Analisis Struktur Data sebelum Pemodelan: Identitas Eksak, Validasi yang Meniru Data Uji, dan Ensemble 41 Model dengan Bobot Adaptif

**DICE FEST 2026 Machine Learning · Solution Writeup · Private LB peringkat 8 (0,24844) · Public LB 0,24913**

**Tim:** BISMILLAH
**Anggota:** [isi nama anggota]
**Afiliasi:** [isi afiliasi]
**Cakupan:** sesuai ketentuan panitia, utas ini hanya memuat bagian 1 sampai 3 (pemahaman masalah dan data, rekayasa fitur, arsitektur model dan hasil validasi). Rekomendasi kebijakan tidak dicantumkan.

---

## 1. Pemahaman Masalah dan Struktur Data

### 1.1 Tugas dan metrik
Lima besaran energi diprediksi untuk setiap kawasan permukiman: gas total, listrik total, intensitas gas per meter persegi (EUI), porsi bangunan bersih, dan adopsi pemanas listrik. Data latih berisi 429.303 kawasan dan data uji 183.987 kawasan, dengan 201 kolom fitur yang namanya disamarkan. Metrik adalah rata-rata RMSE per target yang dinormalisasi dengan σ tetap (151.750,04; 41.725,37; 40,65; 12,49; 7,18). Karena setiap target dinilai terpisah, setiap target boleh ditangani dengan pendekatan berbeda. Pada model akhir, EUI menyumbang sekitar separuh total galat, sehingga sebagian besar upaya pemodelan dicurahkan ke target ini.

### 1.2 Temuan struktur data sebelum model dilatih
Analisis eksploratif difokuskan pada temuan yang mengubah desain solusi.

1. **Gas = EUI × `bld_dim_06`.** Hasil kali EUI dan `bld_dim_06` selalu sama dengan gas total hingga presisi mesin (selisih relatif terbesar 7 × 10⁻¹⁶). Kolom tersebut adalah luas lantai, sehingga gas dan EUI dilayani oleh satu kelompok model.
2. **Porsi bangunan bersih = round(100 × `bld_dim_02` / `bld_dim_01`, 2)** tepat pada seluruh baris latih. Target ini dihitung dengan rumus dan galatnya nol.
3. **Adopsi pemanas listrik seragam dalam satu subzona.** Subzona berperilaku sebagai unit sensus: 113 dari 132 kolom sensus juga konstan per subzona. Sebanyak 92% kawasan uji berada di subzona yang muncul di data latih, sehingga nilai adopsinya dapat disalin langsung.
4. **Data uji diambil secara acak per baris.** Porsi data uji tepat 30%, kawasan uji di subzona baru 8,14% (nilai harapan pengacakan per baris 8,17%), kawasan uji tersebar merata di semua zona, dan validasi adversarial menghasilkan AUC 0,4997.
5. **Batas akurasi.** Varians galat logaritmik mengikuti a + b/M dengan M jumlah meter. Sekitar tiga perempat galat gas dan listrik berasal dari variasi antarrumah tangga yang tidak tercatat dalam data, sehingga perbaikan lebih banyak datang dari pengurangan galat acak model (*ensemble*) daripada dari model yang lebih besar.
6. **Batas nilai.** Gas pada data latih tidak pernah melebihi 999.765 kWh, dan kawasan dengan 2 sampai 4 meter berperilaku berbeda dari kawasan lain.

**[Gambar 1: unggah `reports/figures/fig2_noise_floor.png` di sini, galat terhadap jumlah meter]**

---

## 2. Rekayasa Fitur dan Pra-pemrosesan

Rekayasa fitur dipandu satu pertanyaan: faktor apa yang membuat satu kawasan memakai energi lebih banyak daripada kawasan lain?

- **Ukuran rumah, bukan ukuran kawasan.** Rasio yang menormalisasi besaran kawasan: luas lantai per meter (proksi ukuran rata-rata rumah), luas per sertifikat energi, meter per sertifikat, dan sisa porsi jenis pemanas (100 dikurangi jumlah kategori pemanas yang tercatat).
- **Target per rumah tangga.** Model memprediksi gas per meter, gas per meter persegi, gas per sertifikat, atau logaritmanya, lalu hasilnya dikalikan kembali. Bobot baris diatur agar fungsi kerugian setara dengan galat kuadrat pada skala asli. Sebagian model memakai objektif Poisson (perbaikan EUI model tunggal 0,0039).
- **Rata-rata target wilayah secara *out-of-fold*.** Subzona, zona, petak iklim (kombinasi tiga kolom iklim yang membentuk kisi dengan sekitar lima belas kawasan per petak), dan kawasan dengan nomor subzona berdekatan. Nomor subzona ternyata berurutan secara geografis: bila seluruh kawasan diurutkan menurut nomor subzona, 99,07% kawasan berada di zona yang sama dengan kawasan sebelumnya (96,5% pada tingkat pasangan subzona).
- **Kawasan termirip (kNN-target).** Untuk setiap kawasan dicari kawasan dalam radius geografis tertentu yang stok bangunannya paling mirip (ukuran rumah, cakupan sertifikat, komposisi periode bangunan, porsi bangunan bersih), lalu rata-rata konsumsinya dipakai sebagai fitur, juga secara *out-of-fold*. Versi lanjutannya memakai beberapa definisi kemiripan. Ini perbaikan terbesar dari satu kelompok fitur: galat EUI model tunggal turun 0,0022, konsisten di semua fold.
- **Agregat kawasan.** Jumlah kawasan, total luas, jumlah meter, dan jumlah sertifikat per subzona dan zona, dihitung dari gabungan data latih dan uji tanpa target.
- **Jumlah rumah tangga untuk adopsi.** Kolom komposisi pemanas adalah pecahan dari jumlah rumah tangga. Bilangan bulat terkecil yang membuat semua pecahan bulat adalah jumlah rumah tangga subzona (N), dan K = N × (100 − jumlah porsi tercatat)/100 adalah rumah tangga di luar kategori tercatat.
- **Kontrol kebocoran.** Semua statistik yang memakai target dihitung *out-of-fold*, file fold dibuat sekali dan tidak diubah, seed dikunci, dan ketiadaan kebocoran diuji otomatis.

**[Gambar 2: unggah `reports/figures/fig5_gains.png` di sini, perbaikan dari setiap gagasan]**

---

## 3. Arsitektur Model dan Hasil Validasi

### 3.1 Skema validasi
Karena data uji diambil acak per baris, validasi memakai 10-fold acak dengan file fold tetap. Pemilihan skema ini berpengaruh besar: pada pembagian per zona, rata-rata target wilayah tampak merusak karena baris validasi tidak punya data wilayah, padahal 92% kawasan uji berada di subzona yang dikenal.

| Skema | Rata-rata target wilayah | Gas | EUI | Listrik |
|---|---|---|---|---|
| acak per baris (sesuai data uji) | tanpa | 0,2573 | 0,6721 | 0,3262 |
| acak per baris (sesuai data uji) | dengan | 0,2562 | 0,6699 | 0,3255 |
| per zona | tanpa | 0,2708 | 0,6923 | 0,3302 |
| per zona | dengan | 0,3637 | 0,8331 | 0,3474 |

Tahap setelah penggabungan (koreksi galat dan penyesuaian akhir) dinilai pada *holdout set* 30% data latih. Seluruh model dilatih ulang tanpa bagian ini, dan setiap usulan hanya dinilai padanya. Pada setiap langkah, perbaikan yang diperkirakan *holdout set* hampir sama dengan perubahan skor *leaderboard*, sehingga tidak ada keputusan yang diambil dari *leaderboard* publik.

### 3.2 Tiga jalur untuk lima target
- **Gas dan EUI:** satu kelompok model. Saat penggabungan, bobot kolom gas menekankan kawasan besar, sedangkan bobot kolom EUI tidak.
- **Listrik:** kelompok model tersendiri.
- **Porsi bangunan bersih dan adopsi:** rumus eksak dan penyalinan nilai subzona. Untuk 8% kawasan uji di subzona baru, LightGBM dengan objektif *binomial cross-entropy* berbobot K memprediksi peluang q bahwa satu dari K rumah tangga tidak memakai pemanas listrik, lalu adopsi = 100 × K × (1 − q)/N, dirata-rata dengan regresi langsung.

**[Gambar 3: unggah `reports/figures/fig_arch.png` di sini, alur lengkap dari data mentah hingga berkas submisi]**

### 3.3 Ensemble 41 model
| Kelompok | Model gas/EUI | Model listrik | Galat terbaik EUI (gas) | Galat terbaik listrik | Bobot gas / EUI / listrik |
|---|---|---|---|---|---|
| XGBoost | 16 | 3 | 0,6564 (0,2486) | 0,3200 | 42% / 42% / 16% |
| LightGBM | 5 | 4 | 0,6566 (0,2488) | 0,3197 | 13% / 20% / 23% |
| MLP standar | 2 | 1 | 0,6677 (0,2522) | 0,3213 | 5% / 1% / 3% |
| MLP PLR dan TabM-mini | 3 | 2 | 0,6584 (0,2485) | 0,3188 | 27% / 19% / 25% |
| MLP gas + listrik | 1 | 2 | 0,6585 (0,2488) | 0,3186 | 6% / 11% / 26% |
| NCA (retrieval) | 1 | 1 | 0,6673 (0,2515) | 0,3217 | 8% / 7% / 8% |

NCA lebih lemah sebagai model tunggal, tetapi pola galatnya paling berbeda sehingga berguna dalam *ensemble*. Penggabungan memperbaiki galat dibandingkan model tunggal terbaik sebesar 0,0083 untuk EUI, 0,0036 untuk gas, dan 0,0027 untuk listrik (diukur pada *holdout set*).

### 3.4 Penggabungan, koreksi, dan penyesuaian akhir
1. **NNLS dengan bobot adaptif.** Bobot tak negatif yang berubah secara halus menurut luas kawasan, jumlah meter, dan besar prediksi, karena model yang unggul di kawasan kecil tidak selalu unggul di kawasan besar (LB −0,00009).
2. **Koreksi galat wilayah.** Prediksi dikoreksi sebesar sebagian dari rata-rata galat tetangga pada tingkat zona, petak iklim, nomor subzona berdekatan, dan kawasan termirip. Bobot koreksi ditentukan pada *holdout set*, dan tetangga hanya diambil dari bagian data yang tidak sedang dievaluasi.
3. **Penyesuaian akhir.** Kawasan dengan 2 sampai 3 meter diprediksi sekitar 20% terlalu rendah dan kawasan 4 meter sekitar 3% sampai 7%, sehingga dikoreksi dengan faktor pengali per kelompok. Prediksi gas di dekat batas satu juta kWh dikoreksi dengan pendekatan distribusi normal terpancung, lalu dipotong pada nilai maksimum data latih (LB −0,00007, sama dengan perkiraan *holdout set*).
4. **Kawasan duplikat.** Sebanyak 77 kawasan uji identik dengan kawasan latih (subzona, meter, sertifikat, luas, dan satu kolom sensus yang hampir unik); targetnya disalin dari pasangannya.

### 3.5 Hasil
Galat jujur hasil penggabungan (bobot dihitung di empat per lima data latih, diuji di seperlima sisanya, sebelum koreksi dan penyesuaian akhir): gas 0,2449, listrik 0,3159, EUI 0,6502. Porsi bangunan bersih bergalat nol. Adopsi bergalat nol pada subzona yang dikenal dan RMSE 1,03 poin persentase (0,143 ternormalisasi) pada subzona baru.

| Versi | Langkah | LB publik |
|---|---|---|
| v12 | agregat kawasan + MLP PLR | 0,25025 |
| v15 | koreksi galat wilayah | 0,24954 |
| v16 | fitur kawasan termirip di dalam model | 0,24943 |
| v19 | objektif Poisson + kawasan termirip lanjutan | 0,24934 |
| v19vc | bobot penggabungan adaptif | 0,24925 |
| v19nca | model NCA | 0,24920 |
| final | penyesuaian kawasan bermeter sedikit + batas gas | 0,24913 |

Skor akhir: **public 0,24913, private 0,24844 (peringkat 8)**.

**[Gambar 4: unggah `reports/figures/fig3_progress.png` di sini, perkembangan skor]**

### 3.6 Reprodusibilitas
Seed dikunci di semua pustaka, tidak ada batas waktu pelatihan, dan versi pustaka dicatat. Untuk menguji reprodusibilitas, seluruh 42 model dilatih ulang dari awal: setiap prediksi identik hingga tingkat byte dengan versi pengembangan, dan berkas submisi yang dihasilkan notebook identik dengan berkas yang dipilih (sha256 `aec745ab...d66e`). CatBoost sempat diuji dan hasilnya baik, tetapi tidak dipakai karena implementasi GPU-nya tidak menghasilkan keluaran yang identik di setiap eksekusi.

### 3.7 Yang tidak berhasil
*Stacking* level-2, *ensemble* khusus subzona baru, koreksi galat nonlinear, validasi 20-fold, serta penelusuran kebocoran lewat urutan ID, nilai gas kembar, dan duplikat longgar. Semuanya tercatat di register eksperimen dan tidak memperbaiki skor pada *holdout set*.
