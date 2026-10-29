# Peta Kuadran Zona (Slide 8)

Dokumen ini menjelaskan fungsi *slide* 8, arti kuadran, alasan judul lama membingungkan, dan usulan judul baru beserta naskahnya.

Sumber: teks *slide* 8 di `deck/ppt04/BISMILLAH_Final_v5.pptx`, `final/src/explain.py` baris 149 sampai 174, output notebook In [34], dan `deck/Walkthrough/walkthrough_tahapan.md` Tahap 9a.

## Daftar isi

- [1. Kuadran, bukan kuartil](#1-kuadran-bukan-kuartil)
- [2. Fungsi slide 8](#2-fungsi-slide-8)
- [3. Isi tabel slide 8](#3-isi-tabel-slide-8)
- [4. Mengapa judul lama membingungkan](#4-mengapa-judul-lama-membingungkan)
- [5. Usulan judul](#5-usulan-judul)
- [6. Usulan naskah](#6-usulan-naskah)
- [7. Pertanyaan juri dan jawabannya](#7-pertanyaan-juri-dan-jawabannya)

## 1. Kuadran, bukan kuartil

Kedua istilah ini sering tertukar.

- **Kuartil** membagi data menjadi empat bagian yang sama besar menurut **satu** ukuran.
- **Kuadran** membagi data menurut **dua** ukuran sekaligus. Setiap ukuran dibelah menjadi dua, sehingga terbentuk empat kelompok.

*Slide* 8 memakai kuadran. Karena itu, jumlah zona di setiap kelompok tidak sama: 118, 47, 47, dan 118.

## 2. Fungsi slide 8

*Slide* 8 adalah jembatan dari model ke kebijakan. Isinya tiga hal.

1. **Mengelompokkan 330 zona** menurut dua ukuran:
   - intensitas gas, yaitu konsumsi gas per meter persegi lantai;
   - kenaikan konsumsi listrik tahunan bila semua pemanas gas diganti pompa panas.

   Setiap ukuran dibelah di nilai tengahnya (median 330 zona): 133,5 kWh/m² untuk intensitas gas dan 94,85% untuk kenaikan listrik.
2. **Menunjukkan isi setiap kelompok:** jumlah zona, porsi gas, potensi renovasi, dan kenaikan listrik.
3. **Menetapkan program untuk setiap kelompok.**

| Kuadran | Intensitas gas | Kenaikan listrik | Program |
|---|---|---|---|
| Q1 | di atas median | di atas median | R1 (renovasi) dan R2 (penilaian gardu) |
| Q2 | di atas median | di bawah median | R3 (uji coba pompa panas) |
| Q3 | di bawah median | di atas median | R2 (penilaian gardu) |
| Q4 | di bawah median | di bawah median | tidak ada program khusus |

Kuadran disusun **setelah** model selesai. Nilai kawasan latih memakai data aktual, sedangkan nilai kawasan uji memakai prediksi. Keduanya dijumlahkan per zona, lalu dibelah di median. Kuadran tidak dipakai sebagai fitur model.

## 3. Isi tabel slide 8

| Kuadran | Zona | Porsi gas kota | Potensi renovasi | Kenaikan listrik (rata-rata) |
|---|---|---|---|---|
| Q1 | 118 | 47% | 3,65 TWh | +102% |
| Q2 | 47 | 12% | 1,07 TWh | +90% |
| Q3 | 47 | 15% | 0,86 TWh | +99% |
| Q4 | 118 | 27% | 1,81 TWh | +87% |

Cara membaca setiap kolom:

- **Porsi gas kota:** bagian dari seluruh konsumsi gas kota yang berada di kuadran tersebut. Q1 memuat hampir separuhnya.
- **Potensi renovasi:** perkiraan gas yang dapat berkurang bila karakteristik bangunan yang membuat konsumsi gas di atas rata-rata diturunkan ke rata-rata. Angka ini dihitung dari SHAP. Totalnya 7,39 TWh. Angka ini perkiraan, bukan janji penghematan.
- **Kenaikan listrik:** rata-rata kenaikan konsumsi listrik tahunan bila semua pemanas gas diganti pompa panas.

Angka kenaikan listrik di *slide* 8 adalah **rata-rata**. Tabel di `walkthrough_tahapan.md` Tahap 9a memuat **median** yang sedikit berbeda (102,0; 90,1; 97,3; 88,5). Jangan mencampurkan keduanya saat presentasi.

Dua catatan di bagian bawah *slide*:

- **Asumsi:** 80% gas dipakai untuk pemanasan dan pompa panas menghasilkan 3 kWh panas dari 1 kWh listrik (COP 3). Angka listrik berupa energi tahunan, bukan beban puncak.
- **Uji ketahanan:** urutan zona hampir tidak berubah saat prediksi diberi derau acak (Spearman 0,998). Urutan berubah cukup banyak saat COP dibuat berbeda per iklim (Spearman 0,81, dengan 68 zona berpindah kuadran). Karena itu, daftar zona disebut kandidat.

## 4. Mengapa judul lama membingungkan

Judul lama:

> Peta 330 zona: 118 zona Q1 memuat 47% konsumsi gas, jadi program diurutkan per kuadran

Ada dua masalah.

1. Judul memakai "Q1" dan "kuadran" sebelum artinya dijelaskan.
2. Kata "jadi" menyiratkan bahwa angka 47% menjadi alasan pengurutan program. Padahal, alasannya adalah dua ukuran tadi. Angka 47% hanya menunjukkan bahwa Q1 adalah kelompok terbesar.

## 5. Usulan judul

**Usulan utama**

> Zona dikelompokkan menurut intensitas gas dan kenaikan listrik agar setiap kelompok mendapat program yang sesuai dengan kondisinya

Tujuannya diambil dari pernyataan masalah: urutan program perlu disesuaikan per wilayah.

**Bila judul terlalu panjang**, pisahkan menjadi judul dan subjudul.

> **Judul:** Setiap kelompok zona mendapat program yang sesuai dengan kondisinya
>
> **Subjudul:** Zona dikelompokkan menurut intensitas gas dan kenaikan listrik bila pemanas diganti pompa panas (batas: median 330 zona)

**Alternatif lain**

- "Zona dikelompokkan menurut intensitas gas dan kenaikan listrik agar emisi turun tanpa membebani jaringan." Judul ini menyebut dua sisi trilema, tetapi tidak menyebut sisi biaya.
- "Zona dikelompokkan menurut intensitas gas dan kenaikan listrik agar program dimulai dari zona yang paling tepat." Judul ini paling ringkas, tetapi frasa "paling tepat" kurang spesifik.

**Baris penjelas kuadran** (opsional, di bawah subjudul)

> Q1: gas tinggi, kenaikan listrik tinggi · Q2: gas tinggi, kenaikan listrik rendah · Q3: gas rendah, kenaikan listrik tinggi · Q4: keduanya rendah

Di *slide*, "tinggi" dan "rendah" berarti di atas atau di bawah median. Saat berbicara, lebih aman memakai "di atas median" dan "di bawah median", karena kenaikan listrik Q2 yang disebut "rendah" tetap sekitar +90%.

## 6. Usulan naskah

Sekitar 55 detik.

> "Sekarang kami beralih ke kebijakan. Nilai aktual kawasan latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Zona-zona itu kami kelompokkan menurut dua ukuran: seberapa tinggi intensitas gasnya, dan seberapa besar kenaikan listriknya bila pemanas gas diganti pompa panas. Setiap ukuran dibelah di nilai tengahnya, sehingga terbentuk empat kelompok. Kelompok terbesar adalah Q1, yaitu 118 zona yang memuat 47 persen gas kota. Setiap kelompok mendapat program yang sesuai dengan kondisinya, dan tiga rekomendasi berikut menjelaskan program tersebut. Peta ini tahan terhadap galat prediksi, tetapi peka terhadap asumsi efisiensi pompa panas. Karena itu, daftar zona kami sebut kandidat."

## 7. Pertanyaan juri dan jawabannya

**Mengapa batasnya median?**
Median membagi 330 zona menjadi dua bagian yang sama besar pada setiap ukuran, tanpa memerlukan ambang dari luar data. Kelemahannya, ambang ini tidak memiliki arti fisik, sehingga daftar zona diperlakukan sebagai kandidat.

**Mengapa Q4 tidak mendapat program khusus?**
Intensitas gas dan kenaikan listrik Q4 sama-sama di bawah median, sehingga Q4 bukan prioritas awal. Kalimat ini tidak tertulis di *deck*, jadi siapkan sebagai jawaban lisan.

**Mengapa Q1 dan Q4 jauh lebih banyak daripada Q2 dan Q3?**
Zona dengan intensitas gas tinggi cenderung juga mengalami kenaikan listrik tinggi. Akibatnya, sebagian besar zona berada di Q1 atau Q4.

**Apakah ini peta geografis?**
Bukan. Data tidak memuat koordinat. Letak setiap zona di peta ditentukan oleh dua ukuran tersebut.

**Apakah kuadran dipakai sebagai fitur model?**
Tidak. Kuadran disusun setelah prediksi selesai dan hanya dipakai untuk menyusun kebijakan.

**Seberapa besar pengaruh galat model terhadap peta?**
Sekitar 30% nilai zona berasal dari prediksi. Saat prediksi diberi derau acak, urutan zona hampir tidak berubah (Spearman 0,998, median 8 zona berpindah kuadran).

**Apakah rumah di Q1 lebih kecil?**
Tidak. Rata-rata luas lantai per meteran di Q1 adalah 77,5 m², hampir sama dengan seluruh zona (77,8 m²). Angka ini dihitung dari `reports/policy/zones.csv` dan belum tercantum di notebook maupun *deck*.
