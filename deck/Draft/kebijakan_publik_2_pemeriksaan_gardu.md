# Kebijakan Publik 2: Pemeriksaan Gardu di Zona Q1 dan Q3

Dokumen ini menjelaskan Rekomendasi 2 (R2, *slide* 10) secara utuh: tujuannya, asal angka tambahan listrik, alasan memilih zona Q1 dan Q3, cara menentukan zona yang diperiksa lebih dahulu, langkah pelaksanaan, dan batas-batasnya. Bagian akhir memuat usulan teks *slide* dan naskah.

Sumber: `final/src/explain.py` baris 149 sampai 174, notebook bagian 7.1 sampai 7.3 (terutama sel 51), `deck/ppt01/bukti/cek_kebijakan.json`, `deck/ppt02/bukti/cek_zona_contoh.json`, dan `reports/policy/summary.json`.

## Daftar isi

- [1. Istilah](#1-istilah)
- [2. Isi dan tujuan kebijakan](#2-isi-dan-tujuan-kebijakan)
- [3. Asal angka tambahan listrik](#3-asal-angka-tambahan-listrik)
- [4. Rantai dari data ke kebijakan](#4-rantai-dari-data-ke-kebijakan)
- [5. Temuan SHAP dan posisinya di R2](#5-temuan-shap-dan-posisinya-di-r2)
- [6. Zona yang diperiksa lebih dahulu](#6-zona-yang-diperiksa-lebih-dahulu)
- [7. Asal kata subsidi](#7-asal-kata-subsidi)
- [8. Langkah pelaksanaan](#8-langkah-pelaksanaan)
- [9. Hubungan dengan R1 dan R3](#9-hubungan-dengan-r1-dan-r3)
- [10. Keterbatasan](#10-keterbatasan)
- [11. Pertanyaan juri dan jawabannya](#11-pertanyaan-juri-dan-jawabannya)
- [12. Usulan teks slide 10](#12-usulan-teks-slide-10)
- [13. Usulan naskah](#13-usulan-naskah)

## 1. Istilah

- **Gardu:** tempat tegangan listrik dari jaringan distribusi diturunkan sebelum dialirkan ke rumah. Bila pemakaian listrik di sekitarnya naik melebihi kapasitasnya, gardu dapat kelebihan beban.
- **Tambahan listrik:** listrik yang dibutuhkan bila semua pemanas gas di suatu zona diganti pompa panas. Angka ini hasil hitungan, bukan data dari panitia (bagian 3).
- **Energi tahunan:** total pemakaian listrik selama satu tahun.
- **Beban puncak:** pemakaian listrik tertinggi pada satu waktu, misalnya pagi hari di musim dingin. Gardu kewalahan karena beban puncak, bukan karena total tahunan. Data tidak memuat beban puncak.
- **COP:** jumlah panas yang dihasilkan pompa panas dari 1 kWh listrik. COP 3 berarti 1 kWh listrik menghasilkan 3 kWh panas.

## 2. Isi dan tujuan kebijakan

> **Operator jaringan memeriksa kapasitas gardu di zona Q1 dan Q3 sebelum pompa panas dipasang secara luas, agar jaringan listrik tetap andal.**

| Unsur | Isi |
|---|---|
| Apa | Pemeriksaan kapasitas gardu, lalu penyesuaian bila kapasitasnya kurang |
| Di mana | Zona Q1 dan Q3, dimulai dari REG_263, REG_045, dan REG_270 |
| Kapan | Sebelum pemasangan pompa panas diperluas |
| Oleh siapa | Operator jaringan distribusi bersama pemerintah kota |
| Mengapa | Kenaikan listrik di Q1 dan Q3 di atas median bila pemanas gas diganti pompa panas |
| Tuntutan trilema | Keandalan jaringan |

**Tujuannya** ada tiga:

1. **Mencegah gardu kelebihan beban.** Bila banyak rumah di satu zona beralih ke pompa panas sekaligus, pemakaian listrik dapat naik hampir dua kali lipat.
2. **Memberi waktu bagi operator jaringan** untuk memperkuat gardu atau menyiapkan cara lain sebelum pemakaian listrik naik.
3. **Menentukan urutan pemeriksaan**, sehingga kota tidak perlu memeriksa semua gardu sekaligus.

R2 bukan program subsidi dan bukan perluasan gardu secara otomatis. R2 adalah syarat sebelum pemasangan pompa panas diperluas. Penguatan gardu hanya dilakukan bila hasil pemeriksaan menunjukkan kapasitasnya tidak cukup.

## 3. Asal angka tambahan listrik

Tambahan listrik berasal dari **skenario elektrifikasi penuh** di notebook bagian 7.2. Skenarionya: bila semua pemanas gas di suatu zona diganti pompa panas, berapa listrik tambahan yang dibutuhkan dalam setahun.

**Cara menghitung** (`explain.py` baris 169):

1. Ambil konsumsi gas tahunan setiap zona. Kawasan latih memakai data aktual, sedangkan kawasan uji memakai prediksi.
2. Anggap 80% gas dipakai untuk pemanasan. Sisanya, seperti untuk memasak dan air panas, dianggap tidak ikut diganti.
3. Anggap pompa panas menghasilkan 3 kWh panas dari 1 kWh listrik (COP 3).
4. Tambahan listrik = gas zona × 0,8 ÷ 3.
5. Persentase kenaikan = tambahan listrik ÷ listrik zona saat ini.

Contoh: zona yang memakai 300 GWh gas per tahun membutuhkan tambahan listrik 300 × 0,8 ÷ 3 = 80 GWh per tahun.

**Disebut tahunan** karena data gas dan listrik berupa total pemakaian selama satu tahun.

**Kemunculannya di *deck*:**

| *Slide* | Bentuknya | Fungsinya |
|---|---|---|
| 1 | +94,85% | Di zona tengah, listrik hampir dua kali lipat bila semua pemanas gas diganti pompa panas |
| 8 | Sumbu "kenaikan listrik" | Salah satu dari dua ukuran untuk mengelompokkan zona |
| 10 | Grafik tambahan listrik (GWh) | Menentukan zona yang gardunya diperiksa lebih dahulu |

Di *slide* 1 dan 8 angkanya berupa persentase, sedangkan di *slide* 10 berupa jumlah GWh. Karena itu, *slide* 10 perlu kalimat penghubung (bagian 13).

## 4. Rantai dari data ke kebijakan

1. **Peralihan menaikkan pemakaian listrik.** Pompa panas memakai listrik, sehingga pemakaian listrik naik saat pemanas gas diganti.
2. **Kenaikannya tidak sama di setiap zona.** Median kenaikan 94,85%, dan lima zona tertinggi naik 113% sampai 121%.
3. **Listrik dialirkan melalui gardu.** Zona dengan kenaikan terbesar memiliki risiko kelebihan beban terbesar.
4. **Zona Q1 dan Q3 memiliki kenaikan listrik di atas median**, sehingga gardunya diperiksa sebelum pemasangan pompa panas diperluas.
5. **Urutan pemeriksaan ditentukan oleh jumlah tambahan listrik** dalam GWh, karena urutan ini tetap sama meskipun asumsi COP diubah (bagian 6).

## 5. Temuan SHAP dan posisinya di R2

**Temuan.** Pada model penjelas listrik per rumah, ukuran rumah adalah fitur terkuat (notebook sel 45). Sumbangan SHAP-nya di kawasan rumah besar sekitar 75% lebih tinggi daripada di kawasan rumah kecil.

| Perbandingan | Luas rumah | Selisih |
|---|---|---|
| 5% terbesar dibanding 5% terkecil | ±173 m² dibanding ±37 m² | +74,6% |
| Seperempat terbesar dibanding seperempat terkecil | batas kuartil 94,8 m² dan 61,1 m² | +34,6% |

Angka +74,6% dibulatkan menjadi sekitar 75%. Cara menghitungnya sama dengan +65% pada R1: selisih sumbangan SHAP dikembalikan ke skala asli dengan fungsi eksponensial.

**Arti temuan ini untuk R2.** Beban listrik setiap kawasan sudah berbeda jauh sejak awal, terutama karena ukuran rumah. Karena itu, kesiapan gardu perlu diperiksa per zona, tidak disamaratakan.

**Kelemahan yang perlu diakui.** Hubungan SHAP di R2 adalah yang paling lemah di antara ketiga rekomendasi.
- Angka kenaikan listrik (+94,85% dan +113% sampai +121%) **tidak berasal dari SHAP**. Angka itu hasil hitungan skenario di bagian 3.
- Zona Q1 dan Q3 dipilih dari kenaikan listrik, bukan dari ukuran rumah. Rata-rata luas rumah Q1 (77,5 m²) dan Q3 (82,0 m²) tidak jauh berbeda dari seluruh zona (77,8 m²). Angka ini dihitung dari `reports/policy/zones.csv` dan belum tercantum di notebook maupun *deck*.
- Model penjelas listrik per rumah memiliki R² 0,49, paling rendah di antara model penjelas. Sosio-demografi menyumbang sekitar 40% dari pengaruh model, rasio fisik turunan 26%, dan dimensi bangunan 11% (`summary.json`, kunci `epm`).

## 6. Zona yang diperiksa lebih dahulu

**Dua cara mengurutkan zona:**

- menurut **persentase** kenaikan listrik;
- menurut **jumlah** tambahan listrik dalam GWh.

Kedua urutan ini sangat berbeda (korelasi peringkat Spearman 0,36).

**Uji asumsi.** COP semula dianggap sama di semua zona (COP 3), lalu dibuat berbeda per iklim (2,5 sampai 3,5). Hasilnya:

| Zona | Tambahan listrik, COP 3 | Tambahan listrik, COP per iklim | Peringkat menurut GWh | Peringkat menurut persen |
|---|---|---|---|---|
| REG_263 | 757 GWh | 805 GWh | 1 → 1 | 25 → 51 |
| REG_045 | 442 GWh | 525 GWh | 3 → 3 | 9 → 3 |
| REG_270 | 434 GWh | 514 GWh | 4 → 4 | 5 → 1 |
| REG_290 | | | | 1 → 47 |

- Urutan menurut **persentase** berubah besar. REG_290 turun dari peringkat 1 ke peringkat 47.
- Urutan menurut **jumlah GWh** tidak berubah untuk tiga zona di atas.

**Kesimpulan:** zona yang diperiksa lebih dahulu dipilih menurut jumlah tambahan listrik, karena urutannya tidak bergantung pada asumsi COP.

**Alasan tambahan.** Gardu menanggung jumlah listrik yang mengalir, bukan persentase kenaikannya. Zona kecil bisa mengalami kenaikan persen yang besar, tetapi tambahan bebannya tetap kecil. Alasan ini masuk akal secara teknis, tetapi tidak tertulis di notebook.

**Pengakuan yang perlu disampaikan.** Pada COP 3, tambahan listrik sebanding dengan volume gas zona, sehingga urutan GWh sama dengan urutan volume gas. Stabilitas urutan ini sebagian terjadi karena cara menghitungnya. Yang diklaim stabil hanya tiga zona teratas.

**Ketiga zona berada di Q1.** Lima zona dengan tambahan listrik terbesar maupun lima zona dengan kenaikan persen tertinggi semuanya berada di Q1. Zona Q3 masuk cakupan R2 karena kenaikan listriknya di atas median, tetapi tidak ada zona Q3 di daftar yang diperiksa paling dahulu.

**Pertanyaan terbuka: REG_273.** Zona ini berada di peringkat 2 menurut jumlah tambahan listrik (616 GWh pada COP 3, 707 GWh pada COP per iklim), tetapi tidak termasuk zona yang diperiksa lebih dahulu. Alasannya tidak tertulis di notebook maupun *deck*. REG_273 sudah menjadi zona pertama audit energi di R1. Tim perlu memutuskan salah satu:
- menambahkan REG_273 sebagai zona keempat yang diperiksa lebih dahulu; atau
- menjelaskan bahwa REG_273 sudah ditangani lebih dahulu melalui R1, sehingga kebutuhan panasnya turun sebelum pompa panas dipasang.

Jawaban kedua belum didukung analisis, jadi sebaiknya hanya dipakai bila tim sepakat.

## 7. Asal kata subsidi

Kata "subsidi" pada judul *slide* 10 versi v5 berasal dari notebook bagian 7.3 (sel 51):

- **Butir 2:** "Kesiapan gardu tetap perlu dikonfirmasi dengan operator jaringan sebelum subsidi dibuka."
- **Butir 3:** "Penguatan gardu, *time-of-use tariff* (tarif berbasis waktu pemakaian), dan *heat pump* dengan penyimpan panas perlu direncanakan sebelum subsidi berskala besar dibuka."
- **Butir 4:** "Insentif khusus diperlukan ... misalnya bantuan biaya pemasangan dan pembongkaran *boiler* (ketel pemanas) lama."

Jadi, subsidi yang dimaksud adalah bantuan pompa panas pada butir 4, yang di *deck* menjadi R3. Karena *slide* 10 tampil sebelum *slide* 11, juri mendengar kata "subsidi" sebelum mengetahui subsidi apa. Karena itu, judul dan kotak REKOMENDASI memakai "sebelum pompa panas dipasang secara luas". Hubungannya dengan bantuan di R3 disampaikan secara lisan.

## 8. Langkah pelaksanaan

| Langkah | Yang dilakukan | Tujuannya |
|---|---|---|
| 1. Pemeriksaan awal | Operator jaringan mengukur beban puncak dan kapasitas gardu di REG_263, REG_045, dan REG_270 | Memulai dari tiga zona dengan tambahan listrik terbesar |
| 2. Pemeriksaan lanjutan | Pemeriksaan diteruskan ke zona Q1 dan Q3 lain sesuai urutan tambahan listriknya | Mencakup semua zona yang kenaikan listriknya di atas median |
| 3. Penyesuaian | Bila kapasitas tidak cukup: penguatan gardu, tarif berbasis waktu pemakaian, atau pompa panas dengan penyimpan panas | Menyiapkan jaringan sebelum beban naik |
| 4. Perluasan bertahap | Pemasangan pompa panas diperluas di zona yang gardunya sudah siap | Mencegah gardu kelebihan beban |

Pilihan pada langkah 3 berasal dari notebook bagian 7.3 butir 3. Tarif berbasis waktu pemakaian berarti harga listrik dibuat lebih murah di luar jam sibuk, sehingga sebagian pemakaian bergeser dari waktu puncak.

## 9. Hubungan dengan R1 dan R3

- **R1** menurunkan kebutuhan panas di Q1, sehingga tambahan listrik saat pompa panas dipasang menjadi lebih kecil.
- **R2** memastikan jaringan sanggup menanggung tambahan listrik yang tetap ada.
- **R3** memulai pemasangan pompa panas di Q2, dan langkah pertamanya juga pemeriksaan gardu.

REG_270 dan REG_045 menjadi zona awal di R1 dan R2. Di kedua zona ini, renovasi bangunan dan pemeriksaan gardu berjalan bersama.

## 10. Keterbatasan

- **Energi tahunan, bukan beban puncak.** Data tidak memuat beban puncak maupun kapasitas gardu, sehingga daftar zona hanya menunjukkan urutan pemeriksaan.
- **Angka tambahan listrik bergantung pada dua asumsi** (80% gas untuk pemanasan dan COP 3). Dengan COP per iklim, 68 zona berpindah kuadran (Spearman 0,81).
- **Sekitar 30% nilai zona berasal dari prediksi**, sehingga galat model ikut masuk ke perhitungan.
- **Hubungan SHAP lemah** (bagian 5).
- **Data tidak memuat biaya penguatan gardu**, sehingga anggaran belum dapat dihitung.

## 11. Pertanyaan juri dan jawabannya

**Apa tujuan rekomendasi ini?**
Menjaga keandalan jaringan listrik selama peralihan ke pompa panas. Gardu di zona dengan kenaikan listrik terbesar diperiksa lebih dahulu, sehingga operator punya waktu untuk bersiap.

**Dari mana angka tambahan listrik?**
Dari skenario: konsumsi gas zona dikali 0,8 karena 80% gas dipakai untuk pemanasan, lalu dibagi 3 karena pompa panas menghasilkan tiga kali lipat panas dari listrik yang dipakainya.

**Apa bedanya energi tahunan dengan beban puncak?**
Energi tahunan adalah total listrik selama setahun. Beban puncak adalah pemakaian tertinggi pada satu waktu. Gardu kewalahan karena beban puncak. Data kami tidak memuat beban puncak, sehingga daftar ini hanya menjadi urutan pemeriksaan.

**Apa hubungan SHAP dengan rekomendasi gardu?**
SHAP menunjukkan bahwa beban listrik dasar berbeda jauh antarkawasan, terutama karena ukuran rumah. Karena itu, pemeriksaan gardu dilakukan per zona. Zona yang diperiksa lebih dahulu dipilih dari skenario kenaikan listrik, bukan dari SHAP.

**Mengapa memakai jumlah GWh, bukan persentase?**
Urutan menurut GWh tetap sama meskipun asumsi COP diubah, sedangkan urutan menurut persentase berubah besar. Selain itu, gardu menanggung jumlah listrik, bukan persentase kenaikannya.

**Mengapa urutan GWh stabil?**
Sebagian karena cara menghitungnya. Pada COP 3, tambahan listrik sebanding dengan volume gas zona. Yang kami klaim stabil hanya tiga zona teratas.

**Mengapa REG_273 tidak termasuk?**
Belum ada jawaban tertulis. Lihat bagian 6, "Pertanyaan terbuka".

**Mengapa tidak ada zona Q3 di daftar awal?**
Zona dengan tambahan listrik terbesar semuanya berada di Q1. Zona Q3 tetap diperiksa pada tahap lanjutan karena kenaikan listriknya di atas median.

**Apakah gardu akan diperluas?**
Tidak otomatis. Penguatan hanya dilakukan bila hasil pemeriksaan menunjukkan kapasitasnya tidak cukup. Pilihan lainnya adalah tarif berbasis waktu pemakaian atau pompa panas dengan penyimpan panas.

**Apakah ini kebijakan subsidi?**
Bukan. Rekomendasi ini adalah syarat sebelum bantuan pompa panas (R3) diperluas.

## 12. Usulan teks slide 10

**Judul**
> Rekomendasi 2: pemeriksaan kapasitas gardu di zona Q1 dan Q3 sebelum pompa panas dipasang secara luas

**Judul grafik**
> Tambahan listrik tahunan di tiga zona yang diperiksa lebih dahulu (GWh)

**TEMUAN**
> SHAP: ukuran rumah adalah faktor terkuat untuk konsumsi listrik. Rumah besar diprediksi memakai listrik sekitar 75% lebih banyak daripada rumah kecil.
>
> Bila pemanas gas diganti pompa panas, listrik di lima zona dengan kenaikan tertinggi naik 113–121%, lebih dari dua kali lipat.

**INTERPRETASI**
> Urutan zona menurut jumlah tambahan listrik tetap sama meskipun asumsi efisiensi pompa panas diubah, sedangkan urutan menurut persentase berubah besar. Karena itu, zona yang diperiksa lebih dahulu dipilih menurut jumlah tambahan listrik.

**REKOMENDASI**
> Operator jaringan memeriksa kapasitas gardu di zona Q1 dan Q3 sebelum pompa panas dipasang secara luas, agar jaringan listrik tetap andal.
>
> Data kami menentukan urutan pemeriksaan. Perlu atau tidaknya penguatan gardu diputuskan operator dari pengukuran beban puncak.

**LANGKAH**
> 1. Periksa gardu di tiga zona dengan tambahan listrik terbesar
> 2. Lanjutkan ke zona Q1 dan Q3 lain
> 3. Bila kapasitas kurang: perkuat gardu, terapkan tarif berbasis waktu, atau gunakan pompa panas dengan penyimpan panas
> 4. Perluas pemasangan pompa panas di zona yang gardunya siap

**Pelaksana:** operator jaringan distribusi dan pemerintah kota

**Baris kaki (12 pt)**
> Sumber: notebook bagian 7.1 dan 7.3, In [32], [33], [35]; peringkat dan COP per iklim: skrip cek terpisah

## 13. Usulan naskah

Sekitar 55 detik.

> "Di *slide* pertama kami menyebut listrik zona naik hampir dua kali lipat bila pemanas gas diganti pompa panas. Rekomendasi kedua menjawab risiko itu bagi jaringan listrik. Temuan SHAP menunjukkan bahwa ukuran rumah adalah faktor terkuat untuk konsumsi listrik, sehingga beban setiap kawasan sudah berbeda sejak awal. Di lima zona dengan kenaikan tertinggi, listrik naik lebih dari dua kali lipat. Karena itu, operator jaringan perlu memeriksa kapasitas gardu di zona Q1 dan Q3 sebelum pompa panas dipasang secara luas. Pemeriksaan dimulai dari tiga zona dengan tambahan listrik terbesar, karena urutan ini tetap sama meskipun asumsi efisiensi pompa panas diubah. Data kami hanya memuat energi tahunan, sehingga perlu atau tidaknya penguatan gardu tetap diputuskan operator."
