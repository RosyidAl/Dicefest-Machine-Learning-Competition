# Kebijakan Publik 1: Renovasi Bangunan Lebih Dahulu di Zona Q1

Dokumen ini menjelaskan Rekomendasi 1 (R1, *slide* 9) secara utuh: isi kebijakannya, dari mana asalnya, bagaimana SHAP menghasilkan temuan ukuran rumah, apa hubungan tiga zona di grafik dengan langkah pelaksanaan, dan batas-batasnya. Bagian akhir memuat usulan teks *slide* dan naskah.

Sumber: `final/src/explain.py` baris 149 sampai 174 (perhitungan zona dan indikator), output notebook In [34] dan In [35], `deck/ppt01/bukti/cek_kebijakan.py` baris 163 sampai 176 (perhitungan +65% dan +34%), dan `reports/policy/summary.json`.

## Daftar isi

- [1. Isi kebijakan](#1-isi-kebijakan)
- [2. Rantai dari data ke kebijakan](#2-rantai-dari-data-ke-kebijakan)
- [3. Bagaimana SHAP menemukan pengaruh ukuran rumah](#3-bagaimana-shap-menemukan-pengaruh-ukuran-rumah)
- [4. Indikator potensi renovasi](#4-indikator-potensi-renovasi)
- [5. Hubungan tiga zona di grafik dengan langkah](#5-hubungan-tiga-zona-di-grafik-dengan-langkah)
- [6. Langkah pelaksanaan](#6-langkah-pelaksanaan)
- [7. Keterbatasan](#7-keterbatasan)
- [8. Pertanyaan juri dan jawabannya](#8-pertanyaan-juri-dan-jawabannya)
- [9. Usulan teks slide 9](#9-usulan-teks-slide-9)
- [10. Usulan naskah](#10-usulan-naskah)

## 1. Isi kebijakan

> **Pemerintah kota menjalankan program renovasi bangunan hemat energi di zona Q1, sebelum pemanas di zona itu diganti pompa panas.**

| Unsur | Isi |
|---|---|
| Apa | Program renovasi bangunan: dinding, atap, dan jendela |
| Di mana | 118 zona Q1, dimulai dari REG_273, REG_270, dan REG_045 |
| Urutan | Renovasi lebih dahulu, pemasangan pompa panas kemudian secara bertahap |
| Oleh siapa | Pemerintah kota bersama penyedia jasa renovasi |
| Mengapa | Q1 memuat hampir separuh potensi pengurangan gas (3,65 dari 7,39 TWh) dan mengalami kenaikan listrik tertinggi bila langsung beralih ke pompa panas |
| Ukuran keberhasilan | Konsumsi gas per m² turun pada bangunan yang direnovasi, dibandingkan dengan zona pembanding |
| Tuntutan trilema | Biaya dan emisi |

Notebook (sel 48) menulis "*retrofit* pada dinding, atap, dan jendela". Kata "insulasi" tidak dipakai di notebook, jadi tidak dipakai di sini.

## 2. Rantai dari data ke kebijakan

Rekomendasi ini tersusun dari empat langkah. Tiap langkah memakai bukti yang berbeda.

1. **SHAP menunjukkan bahwa karakteristik bangunan berkaitan dengan gas per m².** Ukuran rumah adalah fitur terkuat (bagian 3). Pada model penjelas gas per m², keluarga fitur bangunan (rasio fisik turunan, era bangunan, dimensi bangunan) menyumbang sekitar 47% dari pengaruh model, sedangkan sosio-demografi sekitar 30% (`summary.json`, kunci `eui`).
2. **Bagian gas yang berkaitan dengan bangunan dihitung per kawasan, lalu dijumlahkan per zona.** Hasilnya indikator potensi renovasi, total 7,39 TWh (bagian 4).
3. **Zona dikelompokkan menjadi empat kuadran** dengan dua ukuran: intensitas gas dan kenaikan listrik bila pemanas gas diganti pompa panas. Q1 berada di atas median pada keduanya. Q1 memuat 47% gas kota dan 3,65 TWh potensi renovasi.
4. **Urutan program.** Karena kenaikan listrik Q1 tertinggi (rata-rata +102%), pompa panas yang langsung dipasang akan menambah beban listrik paling besar. Kebutuhan panas diturunkan lebih dahulu lewat renovasi, sehingga tambahan listrik saat pompa panas dipasang menjadi lebih kecil.

Q1 **tidak** dipilih karena rumahnya kecil. Rata-rata luas lantai per meteran di Q1 77,5 m², hampir sama dengan seluruh zona (77,8 m²). Angka ini dihitung dari `reports/policy/zones.csv` dan belum tercantum di notebook atau *deck*. Temuan ukuran rumah berfungsi sebagai bukti bahwa bangunan berpengaruh terhadap gas, sehingga jenis intervensinya adalah renovasi bangunan.

## 3. Bagaimana SHAP menemukan pengaruh ukuran rumah

**Fitur yang dibaca.** `area_per_meter` = luas lantai kawasan (`bld_dim_06`) dibagi jumlah meteran (`bld_dim_12`). Karena satu meteran kira-kira satu rumah tangga, nilainya mendekati luas satu rumah. Arti kedua kolom ini ditemukan lewat analisis data, karena nama kolom disamarkan panitia.

**Model yang dibaca.** SHAP tidak dihitung pada gabungan 41 model prediksi. SHAP dihitung pada model penjelas terpisah, yaitu satu LightGBM per target yang hanya memakai ciri kawasan terukur. Untuk R1, targetnya log gas per m² (EUI). R² model ini 0,56 pada data uji internal.

**Apa yang dihitung SHAP.** Untuk setiap kawasan, prediksi model dipecah menjadi nilai dasar (rata-rata prediksi) ditambah sumbangan tiap fitur. Sumbangan fitur ukuran rumah:
- positif pada kawasan rumah kecil, sekitar +0,27 pada kelompok 5% terkecil (rata-rata ±37 m²);
- negatif pada kawasan rumah besar, sekitar −0,24 pada kelompok 5% terbesar (rata-rata ±173 m²).

Grafik di *slide* 9 menampilkan kedua titik ini. Nilai tepatnya dihitung `cek_kebijakan.py`.

**Dari selisih SHAP ke +65%.** Model memprediksi log gas per m², sehingga selisih sumbangan harus dikembalikan ke skala asli dengan fungsi eksponensial. Selisih 0,27 − (−0,24) = 0,51, dan e^0,51 ≈ 1,665. Artinya, sumbangan ukuran rumah membuat prediksi gas per m² di ujung rumah kecil sekitar 66,5% lebih tinggi daripada di ujung rumah besar. Notebook membulatkannya menjadi sekitar 65%. Dengan cara yang sama, perbandingan seperempat kawasan terkecil dan seperempat terbesar memberi +34% (tepatnya 33,7%).

**Mengapa model mempelajari pola ini.** Di 429.303 kawasan latih, kawasan dengan luas lantai per meteran yang kecil secara konsisten memiliki gas per m² yang lebih tinggi. Model menangkap pola itu, dan SHAP hanya membaca apa yang dipelajari model. SHAP tidak membuktikan bahwa ukuran rumah menjadi penyebabnya.

**Yang dibandingkan adalah gas per m², bukan total gas.** Total gas rumah besar tetap lebih banyak. Yang lebih tinggi pada rumah kecil adalah gas untuk tiap meter persegi lantai.

**Dugaan penjelasan fisik.** Notebook (sel 48) menduga dinding dan atap rumah kecil relatif luas dibanding ruang yang dipanaskan, sehingga lebih banyak panas hilang per meter persegi lantai. Data tidak memuat luas dinding, luas atap, atau panas yang hilang, jadi penjelasan ini tidak diuji.

## 4. Indikator potensi renovasi

**Definisi** (`explain.py` baris 151 sampai 164). Untuk setiap kawasan, sumbangan SHAP dari tiga keluarga fitur bangunan (era bangunan, dimensi bangunan, rasio fisik turunan) dijumlahkan. Bila jumlahnya positif, artinya karakteristik bangunan membuat gas per m² kawasan itu di atas rata-rata. Indikatornya adalah jumlah gas yang hilang bila bagian positif itu diturunkan ke rata-rata:

indikator kawasan = gas kawasan × (1 − e^(−sumbangan bangunan positif))

Kawasan yang bangunannya membuat gas di bawah rata-rata diberi nilai nol. Indikator kawasan kemudian dijumlahkan per zona.

**Angka hasilnya** (tabel *slide* 8, notebook In [34]):

| Kuadran | Porsi gas kota | Potensi renovasi | Porsi potensi |
|---|---|---|---|
| Q1 | 47% | 3,65 TWh | 49% |
| Q2 | 12% | 1,07 TWh | 14% |
| Q3 | 15% | 0,86 TWh | 12% |
| Q4 | 27% | 1,81 TWh | 24% |
| Total | 100% | 7,39 TWh | 100% |

Kolom "porsi potensi" dihitung dari angka di tabel *slide* 8. Total 7,39 TWh sekitar 5% dari seluruh gas kota (141,35 TWh).

**Cara menyebutnya.** "Indikator" atau "perkiraan", bukan "penghematan". Acuannya rata-rata nasional, dan biasnya bisa ke dua arah.

## 5. Hubungan tiga zona di grafik dengan langkah

Grafik batang di *slide* 9 berjudul "Indikator retrofit tiga zona kandidat (GWh; % gas zona)". Isinya:

| Zona | Indikator potensi | Persen dari gas zona | Kuadran |
|---|---|---|---|
| REG_273 | ±170 GWh | 7,36% | Q1 |
| REG_270 | ±140 GWh | 8,60% | Q1 |
| REG_045 | ±111 GWh | 6,69% | Q1 |

**Mengapa tiga zona ini.** Ketiganya adalah zona dengan indikator potensi renovasi terbesar di seluruh 330 zona, dan ketiganya berada di Q1. Grafik itu menjadi dasar pemilihan titik awal: audit energi dimulai di zona yang potensi pengurangannya terbesar.

**Hubungannya dengan langkah.** Grafik menjawab "di mana memulai". Kotak LANGKAH menjawab "apa yang dilakukan di sana". Keduanya perlu disambungkan dengan satu kalimat, misalnya: "Audit dimulai di tiga zona dengan potensi terbesar, seperti pada grafik."

**Catatan sumber.** Angka 170, 140, dan 111 GWh berasal dari output notebook In [35] (gas zona × persen). `reports/policy/zones.csv` berasal dari putaran yang lebih lama dan memberi angka sedikit berbeda (166, 137, 111 GWh), dengan urutan tiga teratas yang sama. Gunakan angka notebook.

**Irisan dengan R2.** REG_270 dan REG_045 juga menjadi titik awal penilaian gardu di R2. Kedua rekomendasi saling melengkapi di dua zona itu: renovasi menurunkan kebutuhan panas, dan penilaian gardu memeriksa kesiapan jaringan.

## 6. Langkah pelaksanaan

| Langkah | Yang dilakukan | Tujuannya |
|---|---|---|
| 1. Audit energi | Memeriksa bangunan di tiga zona awal untuk melihat bagian mana yang paling banyak kehilangan panas | Memastikan dugaan model di lapangan, karena temuan SHAP hanya asosiasi |
| 2. Renovasi percontohan | Merenovasi sebagian bangunan, dengan zona serupa yang belum direnovasi sebagai pembanding | Memisahkan pengaruh renovasi dari pengaruh lain, misalnya cuaca atau harga energi |
| 3. Pengukuran | Membandingkan gas per m² sebelum dan sesudah renovasi terhadap zona pembanding | Membuktikan apakah renovasi menurunkan konsumsi gas |
| 4. Perluasan | Memperluas program ke zona Q1 lain bila penurunannya terbukti | Menghindari anggaran besar sebelum ada bukti |

Langkah 4 tidak tertulis di *slide* 9 versi v5. Langkah serupa ada di *slide* 11 (R3: "perluas bila terukur"). Menambahkannya membuat ketiga rekomendasi konsisten.

## 7. Keterbatasan

- **Asosiasi, bukan sebab-akibat.** SHAP membaca pola yang dipelajari model dari data observasional.
- **Model penjelas tidak seakurat model prediksi.** R² gas per m² 0,56.
- **Penjelasan fisik tidak diuji.** Data tidak memuat luas dinding, luas atap, jendela, atau panas yang hilang.
- **Indikator bukan janji penghematan.** Acuannya rata-rata nasional.
- **Sekitar 30% nilai zona berasal dari prediksi**, sehingga galat model ikut masuk ke peta.
- **Potensi Q1 besar terutama karena Q1 besar.** Q1 memuat 47% gas dan 49% potensi. Per unit gas, potensi Q2 sedikit lebih tinggi (12% gas, 14% potensi). Q1 didahulukan karena jumlah potensinya terbesar dan kenaikan listriknya tertinggi.
- **Data tidak memuat biaya renovasi**, sehingga program belum dapat dinilai dari sisi anggaran.

## 8. Pertanyaan juri dan jawabannya

**Mengapa rumah kecil lebih boros gas?**
Yang dibandingkan adalah gas per meter persegi, bukan total gas. Dugaan kami, rumah kecil memiliki dinding dan atap yang relatif luas untuk ruang yang dipanaskan, sehingga lebih banyak panas hilang. Data tidak mengukur hal itu secara langsung, jadi ini asosiasi.

**Apakah rumah di Q1 kecil?**
Tidak. Rata-rata luas per meteran di Q1 77,5 m², hampir sama dengan seluruh zona (77,8 m²). Q1 dipilih dari intensitas gas dan kenaikan listrik. Temuan ukuran rumah menunjukkan bahwa bangunan berpengaruh, sehingga intervensinya renovasi.

**65% itu antara apa dan apa?**
Antara 5% kawasan dengan rumah terkecil (±37 m²) dan 5% dengan rumah terbesar (±173 m²). Perbandingan antarkuartil memberi +34%.

**Mengapa tiga zona itu?**
Ketiganya memiliki indikator potensi renovasi terbesar di seluruh kota, dan ketiganya berada di Q1.

**Dari mana 3,65 TWh?**
Dari sumbangan SHAP fitur bangunan per kawasan: gas yang hilang bila karakteristik bangunan yang membuat gas di atas rata-rata diturunkan ke rata-rata, dijumlahkan per zona. Ini indikator, bukan janji penghematan.

**Mengapa Q2 tidak didahulukan untuk renovasi, padahal potensinya per unit gas sedikit lebih tinggi?**
Kenaikan listrik Q2 di bawah median, sehingga pompa panas bisa diuji di sana lebih dahulu (R3). Q1 didahulukan untuk renovasi karena jumlah potensinya terbesar dan kenaikan listriknya tertinggi.

**Bagaimana dengan periode bangunan?**
Kawasan yang seluruh bangunannya berasal dari periode `bld_era_15` memiliki EUI sekitar 7% lebih tinggi. Nama periodenya disamarkan panitia.

## 9. Usulan teks slide 9

**Judul**
> Rekomendasi 1: renovasi bangunan (*retrofit*) lebih dahulu di zona Q1, dimulai dari audit energi

**Judul grafik batang**
> Potensi pengurangan gas terbesar se-kota, ketiganya di Q1 (GWh; % gas zona)

**TEMUAN**
> SHAP: sebagian konsumsi gas berkaitan dengan karakteristik bangunan. Contoh: per m², rumah kecil +65% dibanding rumah besar*

**INTERPRETASI**
> Dijumlahkan per zona: potensi pengurangan gas 7,39 TWh, separuhnya (3,65 TWh) di Q1. Perkiraan dan asosiasi, bukan janji.

**REKOMENDASI**
> Pemerintah kota memprioritaskan renovasi bangunan (dinding, atap, jendela) di zona Q1, sebelum pemanas diganti pompa panas

**LANGKAH**
> 1. Audit energi di tiga zona dengan potensi terbesar
> 2. Renovasi percontohan, dibandingkan dengan zona yang belum direnovasi
> 3. Ukur penurunan gas per m²; perluas ke zona Q1 lain bila terbukti

**Pelaksana:** pemerintah kota dan penyedia jasa renovasi

**Baris kaki (12 pt)**
> *5% rumah terkecil (±37 m²) dibanding 5% terbesar (±173 m²); antarkuartil +34%. Sumber: notebook bagian 7.1, In [33], [34], [35]; +34%: skrip cek terpisah

## 10. Usulan naskah

Sekitar 55 detik. Bila harus tepat 50 detik, kalimat contoh ukuran rumah dapat dihapus karena sudah tertulis di layar.

> "Rekomendasi pertama: pemerintah kota memprioritaskan renovasi bangunan di zona Q1, sebelum pemanas di zona itu diganti pompa panas. Dasarnya dari SHAP model penjelas: sebagian konsumsi gas setiap kawasan berkaitan dengan karakteristik bangunannya. Contohnya, per meter persegi, rumah kecil diprediksi memakai gas sekitar 65 persen lebih banyak daripada rumah besar. Bila dijumlahkan per zona, hampir separuh potensi pengurangan gas, 3,65 terawatt-jam, berada di Q1, dan kenaikan listrik Q1 juga tertinggi. Pelaksanaannya bertahap: audit energi di tiga zona dengan potensi terbesar, renovasi percontohan dengan zona pembanding, lalu diperluas bila penurunan gas terbukti."
