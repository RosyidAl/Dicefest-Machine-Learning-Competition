# Interpretasi Angka Temuan (Slide 8 sampai 11)

Dokumen ini menjelaskan arti tiap angka di bagian kebijakan dengan bahasa sehari-hari: angka itu membandingkan apa dengan apa, bagaimana mengucapkannya, dan apa batasnya. Tujuannya supaya pembicara bisa menyusun narasi sendiri tanpa menghafal istilah teknis.

Sumber angka: teks slide `deck/ppt04/BISMILLAH_Final_v5.pptx`, `deck/ppt04/laporan_v5.md`, dan notebook bagian 7.1 sampai 7.3. Cara membaca: satu angka diucapkan dengan pola "[angka] artinya [A] dibanding [B]". Kalau sebuah angka tidak bisa dijelaskan dalam satu kalimat tanpa istilah teknis, biarkan angka itu hanya tampil di layar.

## Daftar isi

- [Dua asumsi yang dipakai semua angka kenaikan listrik](#dua-asumsi-yang-dipakai-semua-angka-kenaikan-listrik)
- [Slide 8. Peta 330 zona](#slide-8-peta-330-zona)
- [Slide 9. Rekomendasi 1 (retrofit)](#slide-9-rekomendasi-1-retrofit)
- [Slide 10. Rekomendasi 2 (gardu)](#slide-10-rekomendasi-2-gardu)
- [Slide 11. Rekomendasi 3 (pilot pompa panas)](#slide-11-rekomendasi-3-pilot-pompa-panas)
- [Tiga istilah yang cukup dipahami](#tiga-istilah-yang-cukup-dipahami)

## Dua asumsi yang dipakai semua angka kenaikan listrik

Angka kenaikan listrik bukan hasil ukur. Angka itu hasil hitungan dengan dua asumsi:
- **80% gas dipakai untuk pemanasan.** Sisanya (memasak, air panas) dianggap tidak ikut diganti.
- **COP 3.** Satu kWh listrik di pompa panas menghasilkan tiga kWh panas.

Rumusnya: tambahan listrik = gas × 0,8 ÷ 3. Jadi tambahan listrik sekitar 27% dari gas zona. Kedua asumsi ini bukan hasil dari data, dan itu perlu diucapkan bila ditanya.

## Slide 8. Peta 330 zona

| Angka | Artinya | Hati-hati |
|---|---|---|
| **330 zona** | Seluruh kawasan dikelompokkan menjadi 330 zona wilayah. Keputusan program dibuat per zona. | Peta ini bukan peta geografis. Data tidak memuat koordinat. |
| **30% nilai zona dari prediksi** | Nilai sebuah zona adalah jumlah semua kawasannya. Kawasan latih memakai nilai aktual, kawasan uji memakai prediksi, dan prediksi menyumbang sekitar 30%. | Galat prediksi ikut masuk ke peta. |
| **+94,85%** (slide 1) | Di zona yang posisinya di tengah dari 330 zona, listrik tahunan naik sekitar 95% bila semua pemanas gas diganti pompa panas. | Itu skenario dengan dua asumsi di atas, dan energi tahunan, bukan beban puncak. |
| **Q1, Q2, Q3, Q4** | Zona dibagi empat kelompok menurut dua pertanyaan: apakah gasnya lebih boros dari zona tengah, dan apakah kenaikan listriknya lebih besar dari zona tengah. Batasnya median 330 zona. | Q adalah kuadran, bukan kuartil. "Di bawah median" tidak sama dengan "kecil". |
| **118 / 47 / 47 / 118 zona** | Jumlah zona di Q1, Q2, Q3, dan Q4. | Q1 dan Q4 besar karena zona yang boros gas cenderung juga mengalami kenaikan listrik tinggi. |
| **47%, 12%, 15%, 27% gas** | Porsi dari seluruh gas kota yang dipakai tiap kuadran. Q1 memuat hampir separuhnya. | Jumlahnya bisa 101% karena pembulatan. |
| **+102%, +90%, +99%, +87%** | Rata-rata kenaikan listrik per kuadran bila semua pemanas gas diganti. | Ini rata-rata. Tabel median di `walkthrough_tahapan.md` (Tahap 9a) memuat angka berbeda (102,0; 90,1; 97,3; 88,5). Jangan dicampur. |
| **3,65 / 1,07 / 0,86 / 1,81 TWh** | Indikator potensi penghematan gas lewat *retrofit* per kuadran. Totalnya 7,39 TWh, sekitar 5% dari total gas kota 141,35 TWh. | Indikator, bukan janji penghematan. Acuannya rata-rata nasional dan biasnya bisa ke dua arah. |
| **Spearman 0,998** | Urutan zona hampir tidak berubah saat prediksi diberi derau acak. Angka 1 berarti urutan identik. | Hanya menunjukkan tahan terhadap galat model. |
| **Spearman 0,81** | Urutan zona cukup berubah saat COP dibuat berbeda per iklim (68 zona pindah kuadran). | Karena itu daftar zona disebut kandidat, bukan keputusan. |

## Slide 9. Rekomendasi 1 (retrofit)

| Angka | Artinya | Hati-hati |
|---|---|---|
| **±37 m² dan ±173 m²** | Luas lantai rata-rata per rumah pada kelompok 5% kawasan terkecil dan 5% terbesar. | "Per rumah" berarti luas kawasan dibagi jumlah meteran, kira-kira satu meteran satu rumah. |
| **+65%** | Menurut model penjelas, ukuran rumah saja membuat prediksi gas per meter persegi di ujung rumah kecil sekitar 65% lebih tinggi daripada di ujung rumah besar. Bayangkan kotak kecil dan kotak besar: kulit luar kotak kecil relatif lebih luas dibanding isinya. | Perbandingan dua ujung ekstrem. Nilai tepatnya 66,5% di skrip cek (`cek_kebijakan.py`). |
| **+34% (antarkuartil)** | Perbandingan yang sama, tetapi antara seperempat kawasan terkecil dan seperempat terbesar. Lebih moderat dan lebih adil daripada dua ujung ekstrem. | Bila ditanya "65% itu antara apa dengan apa", jawabannya dua ujung ekstrem. Sebut +34% sebagai pembanding. |
| **bld_era_15: +7% EUI** | Kawasan yang seluruh bangunannya berasal dari satu periode tertentu memiliki intensitas gas sekitar 7% lebih tinggi daripada kawasan tanpa bangunan periode itu. | Nama periode disamarkan, jadi tidak diketahui periode mana. Efeknya kecil, jadi aman disimpan sebagai jawaban Q&A. |
| **Q1: 118 zona, 47% gas** | Sasaran *retrofit*: zona paling boros gas dan paling besar kenaikan listriknya. | Lihat slide 8. |
| **3,65 TWh** | Indikator potensi penghematan gas di Q1, sekitar separuh dari 7,39 TWh. | Bukan janji. |
| **REG_273 (±170 GWh), REG_270 (±140 GWh), REG_045 (±111 GWh)** | Tiga zona awal audit energi. Angka GWh adalah indikator potensi *retrofit* tiap zona, dan tanda kurungnya adalah persen terhadap gas zona itu (7,36%; 8,60%; 6,69%). | Tidak perlu menghafal kodenya. Cukup "tiga zona pertama". |

## Slide 10. Rekomendasi 2 (gardu)

| Angka | Artinya | Hati-hati |
|---|---|---|
| **+75%** | Menurut model penjelas, ukuran rumah saja membuat prediksi listrik per rumah di kawasan rumah besar sekitar 75% lebih tinggi daripada di rumah kecil. Rumah besar memang memakai lebih banyak listrik. | Asosiasi, bukan sebab-akibat. |
| **+113% sampai +121%** | Kenaikan listrik tahunan pada lima zona dengan kenaikan persen tertinggi. Di zona itu listrik lebih dari dua kali lipat. | Bandingkan dengan median 94,85%. |
| **757 / 442 / 434 GWh** | Tambahan listrik tahunan di REG_263, REG_045, dan REG_270 pada COP 3. 1 GWh sama dengan satu juta kWh. | Energi tahunan, bukan beban puncak. |
| **805 / 525 / 514 GWh** | Tambahan yang sama bila COP dibuat berbeda per iklim (2,5 sampai 3,5). | Menunjukkan angkanya naik, tetapi ketiga zona tetap teratas. |
| **Peringkat 1→1, 3→3, 4→4** | Peringkat tiga zona itu menurut tambahan GWh tidak berubah antara COP 3 dan COP per iklim. Karena itu titik awal dipilih menurut GWh. | Sebagian karena konstruksi: pada COP 3, tambahan GWh sebanding dengan volume gas zona. |
| **REG_290: 1→47** | Menurut kenaikan persen, REG_290 peringkat 1 pada COP 3, lalu turun ke peringkat 47 pada COP per iklim. Ini alasan peringkat persen tidak dipakai sebagai titik awal. | |

## Slide 11. Rekomendasi 3 (pilot pompa panas)

| Angka | Artinya | Hati-hati |
|---|---|---|
| **9,46%** | Di kawasan latih yang porsi gas jaringannya 60 sampai 80%, rata-rata 9,46% rumah tangga sudah berpemanas listrik. | Adopsi adalah porsi rumah tangga pada satu waktu, bukan "sudah beralih". |
| **2,73% (di atas 80%)** | Di kawasan yang porsi gas jaringannya di atas 80%, hanya 2,73% rumah tangga berpemanas listrik. Kelompok ini mencakup 70% kawasan latih. | Porsi jenis pemanas berjumlah 100%, jadi ini pola komposisi, bukan sebab-akibat. |
| **+4 sampai +12 poin (di bawah 40%)** | Pada kawasan dengan gas jaringan di bawah 40%, faktor ini menaikkan prediksi adopsi sebesar 4 sampai 12 poin persentase. Di atas 80%, sumbangannya negatif. | Di notebook sel 48 butir 4. |
| **REG_033** | Contoh zona Q2: gas jaringan 83,21%, adopsi 4,99%. Zona ini tetap Q2 pada kedua skenario COP. | Tambahan listriknya 185 GWh (peringkat 52 dari 330). "Di bawah median" berlaku untuk persen, bukan GWh. |
| **Q2** | Zona yang gasnya di atas median tetapi kenaikan listriknya di bawah median. | Kenaikan listrik Q2 masih sekitar +90%. |

## Tiga istilah yang cukup dipahami

- **SHAP:** ukuran seberapa besar satu ciri kawasan menggeser prediksi model. Ini asosiasi di dalam model, bukan bukti sebab-akibat.
- **EUI:** gas per meter persegi lantai.
- **COP:** berapa kWh panas yang dihasilkan pompa panas dari satu kWh listrik. Dipakai COP 3 sebagai asumsi.

Hal lain di bagian kebijakan (NNLS, kNN, *out-of-fold*, dan sejenisnya) tidak perlu diucapkan oleh pembicara kebijakan. Bila ditanya, arahkan ke rekan teknis.
