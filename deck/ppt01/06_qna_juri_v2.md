# Tanya Jawab Juri v2: Status Risiko dan Bank Pertanyaan

Dokumen ini melengkapi `../03_qna_juri.md`. Pedoman jawaban rinci P1 sampai P5 di berkas 03 tetap berlaku; yang berubah
adalah status bagian **RISIKO**. Status itu diperbarui dengan bukti yang dijalankan ulang memakai modul notebook dan
dengan koreksi dari audit independen (27 Oktober). Bagian B berisi 18 pertanyaan yang paling mungkin diajukan juri
campuran (data scientist dan pembuat kebijakan), masing-masing dengan jawaban dua sampai tiga kalimat dan slide
rujukannya.

**Cara memakai.** Jawab dengan urutan: inti jawaban, bukti (sebut angka dan nomor slide atau sel), lalu batasannya.
Bila juri meminta bukti, buka slide cadangan yang tertulis di akhir jawaban.

## A. Status bagian RISIKO dari berkas 03

| Risiko di 03 | Status | Bukti | Slide |
|---|---|---|---|
| **P1.** Bobot gabungan dari prediksi *out-of-fold* (satu model per fold) dipakai pada prediksi uji (rata-rata 10 model) | **didukung, belum diuji ulang di ensemble final** | Di *holdout set* P70/H30, prediksi baris H berasal dari rata-rata model fold, sama seperti data uji. Bobot dari prediksi *out-of-fold* memberi 0,25096, setara dengan bobot yang di-fit langsung di H (0,25095). Bukti ini berasal dari ensemble pengembangan 13 Oktober (D002), bukan ensemble final 41 model. Skor privat juga tidak memburuk dibanding publik | 7, C7 |
| **P2.** Skor jujur bobot adaptif `A2xM2xp2` tidak dicetak di notebook | **dihitung ulang** | Dihitung dengan fungsi notebook pada array run final; skor global mereproduksi sel 35 persis. Hasil adaptif: gas 0,24480 (−0,000146), EUI 0,64978 (−0,000453), listrik 0,31585 (−0,000027), rata-rata 5 target −0,000125. Hitungan ini bukan bukti independen, karena bentuknya dipilih di baris yang sama (D012). Bukti independennya adalah *leaderboard* (−0,00009) | C1 |
| **P3.** λ negatif dan konstanta K, WIN, KSHR | **sebagian** | Audit akhir: versi ridge mengubah skor −0,0000027 dengan tanda per fold campur. K, WIN, dan KSHR diperiksa ulang (D015); KSHR tanpa efek karena kumpulan tetangga selalu penuh. Peta sensitivitas lengkap konstanta belum ada | C2 |
| **P4.** Uji kebocoran otomatis tidak ada di notebook | **ditutup untuk fitur kNN-target dan rata-rata wilayah** | Uji label acak pada modul notebook, satu fold. (1) Label 42.931 baris validasi diacak: fitur validasi dan uji identik bit, sedangkan kontrol yang sengaja bocor berubah di 42.931 baris. (2) Label 77.275 baris latih satu bagian dalam diacak: fitur baris itu identik bit. Uji serupa sudah ada sejak pengembangan (`tests/test_no_leakage.py`). Di luar cakupan: koreksi galat wilayah, yang hanya memakai galat baris latih untuk baris uji | 5, C7 |
| **P5.** Asal parameter S1 dan S3 | **sebagian** | D015: S1 dari prediksi *out-of-fold* penuh gabungan v19nca pada kawasan 2 sampai 4 meter; k pada S3 dari *holdout*; batas S4 dari data latih. Gabungan S1, S3, dan S4 di H memberi −0,000065, semua fold negatif; LB −0,00007. Bentuk koreksi lain di luar normal terpotong belum diuji | C3 |

**Sebutkan lebih dulu sebelum juri menemukannya:**
- XGBoost dan LightGBM memakai *early stopping* pada fold validasinya sendiri (sel 36), sehingga galat *out-of-fold*
  sedikit optimistis. Semua keputusan diambil dari *holdout set*.
- Angka "potensi *retrofit*" adalah batas atas (lihat pertanyaan 15).
- Urutan zona peka terhadap COP yang berbeda per iklim (lihat pertanyaan 10).

## B. Bank pertanyaan

### Validasi dan kebocoran (paling sering ditanya data scientist)

1. **"Bagaimana Anda tahu tidak ada kebocoran?"**
   Semua fitur yang memakai target dihitung *out-of-fold* dua lapis, dan kami mengujinya dengan label acak. Ketika
   label 42.931 baris validasi diacak, fitur validasi dan uji tidak berubah satu bit pun. Ketika label 77.275 baris latih
   satu bagian dalam diacak, fitur baris itu juga tidak berubah. Versi yang sengaja dibuat bocor langsung berubah di
   semua baris. (Slide 5, C7)

2. **"Mengapa validasi acak per baris, bukan per wilayah?"**
   Data uji diambil acak per baris. Validasi adversarial tidak bisa membedakan data latih dari data uji (AUC 0,4997),
   dan porsi kawasan uji di subzona baru 8,14% sesuai harapan pengacakan 8,17%. Pada skema per zona, fitur rata-rata
   wilayah tampak merusak: galat EUI naik dari 0,6923 menjadi 0,8331. Itu karena skema ini menguji situasi yang tidak
   terjadi di data uji. (Slide 7, C5)

3. **"Apakah model Anda overfit ke leaderboard publik?"**
   Komponen yang diterima dipilih di *holdout set* 30%, dan *leaderboard* hanya mengonfirmasi arahnya. Satu kali kami
   menolak perubahan karena *leaderboard* tidak mengonfirmasi: model adopsi v3, 0,24938 vs 0,24934 (D011). Skor privat
   0,24844 tidak memburuk dibanding publik 0,24913, dan peringkat naik dari 9 ke 8. Selisih sekecil itu masih dalam
   derau, jadi kami menyebutnya konsisten dengan tidak *overfit*, bukan bukti mutlak. (Slide 7)

### Model dan arsitektur

4. **"Mengapa 41 model, bukan satu model terbaik?"**
   Galat model-model itu berbeda pola, sehingga gabungannya saling menutupi. Di *holdout* versi v16, gabungan
   memperbaiki model tunggal terbaik sebesar 0,0083 untuk EUI, 0,0036 untuk gas, dan 0,0027 untuk listrik (*writeup*
   bab 6). Model besar hanya dipakai untuk tiga kolom. (Slide 6)

5. **"Bobot adaptif terdengar seperti tuning berlebihan."**
   Ada 8 bobot pojok per model: 224 untuk gas, 224 untuk EUI, dan 104 untuk listrik, semuanya tidak negatif, untuk
   429.303 baris. Pada skor jujur, tidak ada kolom yang memburuk dan perbaikan terbesar ada di EUI. Tiga simpul tidak
   lebih baik daripada dua, jadi kami memakai bentuk paling sederhana. (C1)

6. **"Apa yang sudah dicoba tetapi gagal?"**
   Ada empat: *stacking* model lapis kedua, *ensemble* khusus subzona baru, koreksi galat nonlinear, dan validasi 20 fold.
   Setelah kompetisi kami juga menguji kaskade lintas target gas dan listrik; perbaikannya hanya −0,0000055. Semua
   tercatat di register eksperimen. (*writeup* bab 6)

7. **"Apakah hasilnya bisa direproduksi?"**
   Bisa. *Seed* dikunci, pustaka berjalan deterministik, dan *thread* BLAS dibatasi satu. Sidik sha256 berkas submisi
   tercetak di sel 35 dan sama dengan berkas yang dipilih bila dijalankan di perangkat yang sama. Pelatihan penuh
   sekitar 20 jam; tanpa pelatihan sekitar 5 jam (sel 0). (Slide 6, C3)

### Interpretasi dan kebijakan (paling sering ditanya pembuat kebijakan)

8. **"Apakah SHAP membuktikan sebab-akibat?"**
   Tidak. SHAP menunjukkan sumbangan variabel pada prediksi model. Karena itu setiap rekomendasi kami usulkan sebagai
   pilot dengan zona pembanding, supaya dampaknya terukur sebelum diperluas. (Slide 8, 13)

9. **"Mengapa SHAP dari model penjelas terpisah, bukan dari gabungan 41 model?"**
   Gabungan memakai rata-rata target wilayah yang bukan tuas kebijakan dan sulit dibaca. Model penjelas hanya memakai
   karakteristik terukur. Kecocokannya kami laporkan: R² 0,563 untuk EUI dan 0,491 untuk listrik per meter (sel 41).
   (C8)

10. **"Seberapa yakin dengan urutan zona dan angka kenaikan listrik?"**
    - **Angka:** energi tahunan dengan asumsi 80% gas untuk pemanasan dan COP 3. Asumsi seragam lain hanya
      menskalakan angka, antara 71% dan 128%.
    - **Ketahanan urutan:** kami menguji dua hal yang bisa mengubah urutan.
      - Galat prediksi hampir tidak berpengaruh: Spearman 0,998, lima zona teratas tetap.
      - COP yang berbeda per iklim, dari 2,5 sampai 3,5, berpengaruh besar: Spearman 0,81, 68 zona pindah kuadran,
        dan hanya 1 dari 5 zona teratas tetap.
    - **Kesimpulan:** daftar zona adalah kandidat, dan operator menilai dengan data beban dan iklim lokal. (Slide 9, C4)

11. **"Siapa yang menjalankan rekomendasi, dan data apa yang masih dibutuhkan?"**
    - *Retrofit*: pemerintah kota bersama penyedia jasa.
    - Penilaian gardu: operator jaringan distribusi.
    - Insentif pompa panas: pemerintah kota bersama penyedia.
    - Data yang dibutuhkan sebelum eksekusi: beban puncak, kapasitas gardu, dan COP lokal per zona.

    (Slide 10 sampai 12, 14)

12. **"Apakah adopsi pemanas listrik hanya ada di kawasan yang tidak tersambung gas?"**
    Tidak. Yang berbeda adalah tingkatnya: sekitar 40% di kawasan bergas di bawah 40%, dan hanya 2,73% di kawasan di atas
    80%. Karena kawasan bergas di atas 80% mencakup 70% kawasan, sebagian besar pengguna pemanas listrik justru tinggal
    di kawasan bergas 60% ke atas. Porsi jenis pemanas berjumlah 100%, jadi kami membaca ini sebagai pola, bukan tombol;
    insentifnya dijalankan sebagai pilot. (Slide 12)

### Data

13. **"Nama kolom disamarkan. Bagaimana Anda menafsirkannya?"**
    Kami menduga maknanya dari identitas yang tepat di seluruh data. Contohnya, EUI dikali `bld_dim_06` selalu sama
    dengan gas, jadi `bld_dim_06` adalah luas lantai. Contoh lain, kolom `hvac_dist` adalah pecahan jumlah rumah tangga.
    Setiap tafsiran ditulis sebagai dugaan, termasuk `bld_era_15` sebagai periode bangunan. (Slide 3, 8)

14. **"Apa keterbatasan terbesar data ini?"**
    Ada dua. Pertama, untuk gas dan listrik, sebagian besar galat yang tersisa berasal dari variasi antarrumah tangga
    yang tidak tercatat (kurva a + b/M di *writeup*). Kedua, data tidak memuat beban puncak, kapasitas jaringan, dan
    COP lokal. Karena itu kami memegang urutan zona sebagai kandidat, bukan angka absolutnya. (Slide 13)

### Pertanyaan tambahan dari audit independen

15. **"Apakah 7,39 TWh potensi retrofit itu realistis?"**
    Angka itu batas atas. Definisinya di sel 48: sumbangan SHAP stok bangunan di atas rata-rata, yaitu periode bangunan,
    dimensi bangunan, dan rasio fisik. Kelompok itu ikut memuat ukuran rumah, yang tidak bisa diubah oleh *retrofit*.
    Bila hanya periode bangunan yang dihitung, sisanya sekitar separuh. Karena itu rekomendasi 1 dimulai dari audit
    energi di zona pilot. (Slide 9, 10, 13)

16. **"Mengapa rekomendasi 2 dimulai dari REG_290, padahal tambahan listrik absolutnya bukan yang terbesar?"**
    Persentase mengukur tekanan relatif terhadap beban dan kapasitas yang ada sekarang, sedangkan angka absolut mengukur
    besar investasi. REG_290 naik 121% secara relatif, tetapi tambahan absolutnya peringkat 46. Tambahan absolut terbesar
    ada di REG_263 (0,76 TWh), lalu REG_273, REG_045, dan REG_270, yang juga kandidat *retrofit*. Keduanya perlu dinilai
    operator. (Slide 11)

17. **"Zona Q2, Q3, dan Q4 diapakan?"**
    - Q1 mendapat rekomendasi 1 dan 2: *retrofit* dulu dan cek gardu.
    - Q2 menjadi tempat pilot rekomendasi 3, karena intensitas gasnya tinggi tetapi kenaikan listriknya relatif lebih
      rendah. Contohnya REG_249: gas jaringan 85,54%, adopsi 3,75%.
    - Q3 mengikuti rekomendasi 2: siapkan jaringan dulu.
    - Q4 mendapat program standar.

    (Slide 9, tabel kolom "Program")

18. **"Bukankah +65% itu perbandingan dua ujung yang ekstrem?"**
    Ya. Angka itu membandingkan 5% kawasan dengan rumah terkecil (rata-rata ±37 m² per meter) dan 5% terbesar
    (±173 m²). Antara kuartil bawah dan atas, selisihnya sekitar +34% untuk gas per m² dan +35% untuk listrik per rumah.
    Arah dan bentuknya sama, besarnya lebih kecil. (Slide 8)

## C. Pernyataan tentang alat bantu AI

Dipakai bila juri bertanya langsung (sama dengan berkas 03):

> "Kami memakai Claude sebagai alat bantu pemrograman: menulis dan merapikan kode, menjalankan eksperimen, dan
> menyusun dokumentasi. Semua nilai prediksi dihasilkan oleh kode open-source yang dijalankan di komputer kami, bukan
> oleh model bahasa. Setiap komponen hanya dipertahankan bila memperbaiki skor validasi, dan alasan desainnya bisa kami
> jelaskan."

## D. Pembagian penjawab (tim tiga orang)

- **Orang 1 (jangkar).** Mengulang inti pertanyaan dalam satu kalimat, menyebut siapa yang menjawab, dan membuka slide
  cadangan. Menjawab sendiri pertanyaan umum dan bagian C.
- **Orang 2 (teknis).** Menjawab pertanyaan 1 sampai 7 dan 13; cadangan C1, C2, C3, C5, C7.
- **Orang 3 (kebijakan).** Menjawab pertanyaan 8 sampai 12 dan 14 sampai 18; cadangan C4, C6, C8.
- **Bila ragu,** jawab inti dalam satu kalimat, sebut slide cadangannya, lalu buka slidenya. Di mode slideshow
  PowerPoint, ketik nomor slide lalu Enter: C1 = 15, C2 = 16, C3 = 17, C4 = 18, C5 = 19, C6 = 20, C7 = 21, C8 = 22.
