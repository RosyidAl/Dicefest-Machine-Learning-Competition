# Kebijakan Publik 3: Uji Coba Pompa Panas di Zona Q2

Dokumen ini menjelaskan Rekomendasi 3 (R3, *slide* 11) secara utuh: isi kebijakannya, temuan yang mendasarinya, alasan memilih zona Q2 dan REG_033, langkah pelaksanaan, dan batas-batasnya. Bagian akhir memuat usulan teks *slide* dan naskah.

Sumber: `deck/ppt01/bukti/cek_kebijakan.py` baris 44 sampai 52 (pemakaian pemanas listrik per kelompok gas pipa), `deck/ppt02/bukti/cek_zona_contoh.json` (zona Q2 yang tetap Q2), `reports/policy/summary.json` (model penjelas adopsi), dan notebook bagian 7.1 sampai 7.3.

## Daftar isi

- [1. Istilah](#1-istilah)
- [2. Isi kebijakan](#2-isi-kebijakan)
- [3. Rantai dari data ke kebijakan](#3-rantai-dari-data-ke-kebijakan)
- [4. Temuan: pemakaian pemanas listrik menurut porsi gas pipa](#4-temuan-pemakaian-pemanas-listrik-menurut-porsi-gas-pipa)
- [5. Mengapa disebut pola komposisi](#5-mengapa-disebut-pola-komposisi)
- [6. Mengapa zona Q2 dan mengapa REG_033](#6-mengapa-zona-q2-dan-mengapa-reg_033)
- [7. Langkah pelaksanaan](#7-langkah-pelaksanaan)
- [8. Keterbatasan](#8-keterbatasan)
- [9. Pertanyaan juri dan jawabannya](#9-pertanyaan-juri-dan-jawabannya)
- [10. Usulan teks slide 11](#10-usulan-teks-slide-11)
- [11. Usulan naskah](#11-usulan-naskah)

## 1. Istilah

- **Gas pipa:** gas alam yang dialirkan ke rumah melalui pipa, bukan dari tabung. Notebook menyebutnya "gas jaringan" (kolom `hvac_dist_02`, porsi pemanas gas jaringan di kawasan). Di *slide* dipakai "gas pipa" karena lebih dikenal.
- **Adopsi:** porsi rumah tangga yang memakai pemanas listrik. Angka ini menggambarkan keadaan pada satu waktu, bukan perubahan dari waktu ke waktu. Karena itu, kata yang tepat adalah "memakai", bukan "beralih".
- **Kawasan dan zona:** kawasan adalah satu baris data, rata-rata berisi sekitar 22 rumah tangga. Zona adalah kelompok kawasan dengan kode REG_xxx. Temuan dibaca di tingkat kawasan, sedangkan program ditetapkan per zona.

## 2. Isi kebijakan

> **Pemerintah kota memulai uji coba pompa panas di zona Q2, termasuk bantuan pemasangan dan pembongkaran pemanas gas lama.**

| Unsur | Isi |
|---|---|
| Apa | Uji coba pompa panas, disertai bantuan pemasangan dan pembongkaran pemanas gas lama |
| Di mana | Zona Q2, dengan contoh REG_033 |
| Oleh siapa | Pemerintah kota bersama penyedia pompa panas |
| Mengapa | Kawasan dengan gas pipa tinggi hampir tidak memakai pemanas listrik. Di antara zona bergas, Q2 memiliki kenaikan listrik di bawah median, sehingga uji coba dapat dimulai dengan risiko jaringan yang lebih rendah |
| Ukuran keberhasilan | Pemakaian pemanas listrik naik di zona uji coba, dibandingkan dengan zona pembanding |
| Tuntutan trilema | Transisi yang terukur |

## 3. Rantai dari data ke kebijakan

1. **Temuan.** Di kawasan yang lebih dari 80% rumahnya memakai gas pipa, hanya 2,73% rumah yang memakai pemanas listrik. Kelompok ini mencakup 70% kawasan, sehingga sebagian besar kota tertinggal dalam elektrifikasi pemanas.
2. **Kesimpulan.** Agar transisi berjalan, program harus menyasar kawasan bergas, karena di sanalah sebagian besar rumah berada.
3. **Pertimbangan risiko.** Mengganti pemanas di semua kawasan bergas sekaligus dapat membebani jaringan listrik. Karena itu, program dimulai dari zona bergas yang kenaikan listriknya lebih rendah, yaitu Q2.
4. **Bentuk program.** Rumah di zona ini sudah memiliki pemanas gas yang masih berfungsi. Bantuan pembongkaran pemanas lama mengurangi biaya peralihan bagi warga. Alasan ini merupakan pertimbangan kebijakan, bukan hasil dari data.
5. **Uji coba lebih dahulu.** Temuan di atas berupa asosiasi, sehingga program dimulai dengan uji coba yang dibandingkan dengan zona pembanding.

## 4. Temuan: pemakaian pemanas listrik menurut porsi gas pipa

Angka berikut dihitung dari 429.303 kawasan latih. Pemakaian pemanas listrik dihitung sebagai porsi rumah tangga, dengan jumlah meteran sebagai bobot.

| Porsi gas pipa di kawasan | Porsi dari seluruh kawasan | Rumah tangga yang memakai pemanas listrik |
|---|---|---|
| 40% atau kurang | 0,55% | 40,24% |
| 40–60% | 3,04% | 26,92% |
| 60–80% | 26,32% | 9,46% |
| Lebih dari 80% | 70,08% | 2,73% |

**Mengapa *slide* membandingkan 2,73% dengan 9,46%, bukan dengan 40,24%.** Kelompok dengan gas pipa 40% atau kurang hanya mencakup 0,55% kawasan. Perbandingan dengan kelompok sekecil itu akan berlebihan. Kelompok 60–80% mencakup 26,32% kawasan, sehingga perbandingannya lebih wajar.

**Hasil SHAP.** Model penjelas adopsi menunjukkan pola yang sama. Pada kawasan dengan gas pipa di bawah 40%, porsi gas pipa menaikkan prediksi adopsi sebesar 4 sampai 12 poin persentase. Pada kawasan dengan gas pipa di atas 80%, sumbangannya negatif (notebook sel 48 butir 4).

**Model penjelas adopsi.** R² model ini 0,99. Keluarga fitur bauran pemanas menyumbang 81% dari pengaruh model, sedangkan sosio-demografi 15% (`summary.json`, kunci `adopt`). R² yang sangat tinggi ini wajar, karena porsi jenis pemanas saling melengkapi hingga 100% (bagian 5). Angka ini tidak menunjukkan bahwa model memahami perilaku warga.

## 5. Mengapa disebut pola komposisi

Porsi jenis pemanas di satu kawasan berjumlah 100%. Bila lebih dari 80% rumah memakai gas pipa, porsi yang tersisa untuk pemanas listrik paling banyak di bawah 20%. Akibatnya, adopsi yang rendah di kawasan bergas sebagian terjadi dengan sendirinya.

Karena itu, temuan ini menjawab **di mana** elektrifikasi belum terjadi, bukan **mengapa** belum terjadi. Temuan ini tidak membuktikan bahwa gas pipa menghambat peralihan ke pemanas listrik.

## 6. Mengapa zona Q2 dan mengapa REG_033

**Zona Q2** memiliki intensitas gas di atas median dan kenaikan listrik di bawah median.

| Unsur | Q2 |
|---|---|
| Jumlah zona | 47 |
| Porsi gas kota | 12% |
| Kenaikan listrik rata-rata | +90% |
| Zona yang tetap Q2 pada kedua skenario COP | 33 dari 47 |

Intensitas gas yang tinggi berarti emisi yang dapat diturunkan besar. Kenaikan listrik di bawah median berarti tambahan beban jaringan relatif lebih rendah. Perlu diingat, +90% tetap kenaikan yang besar. Karena itu, langkah pertama uji coba adalah pemeriksaan gardu.

**REG_033** dipakai sebagai contoh karena:

- zona ini tetap berada di Q2 pada COP 3 maupun pada COP yang berbeda per iklim;
- 83,21% rumahnya memakai gas pipa, sehingga termasuk kelompok kawasan bergas pada temuan;
- baru 4,99% rumahnya yang memakai pemanas listrik;
- kenaikan listriknya 84,92% pada COP 3 dan 78,37% pada COP per iklim, lebih rendah daripada REG_242 dan REG_245, dua zona Q2 lain yang konsumsi gasnya lebih besar.

Alasan memilih REG_033 dibandingkan zona Q2 lain tidak tertulis di notebook. Perbandingan di butir terakhir dihitung dari `cek_zona_contoh.json`. Bila ditanya, sampaikan bahwa REG_033 adalah salah satu contoh dari 33 zona Q2 yang posisinya stabil.

Tambahan listrik REG_033 sekitar 185 GWh per tahun (peringkat 52 dari 330). Kata "di bawah median" berlaku untuk persentase kenaikan, bukan untuk jumlah absolutnya.

## 7. Langkah pelaksanaan

| Langkah | Yang dilakukan | Tujuannya |
|---|---|---|
| 1. Pemeriksaan gardu | Operator jaringan memeriksa kapasitas gardu di zona uji coba | Memastikan jaringan sanggup menampung tambahan listrik, karena kenaikan listrik Q2 tetap sekitar +90% |
| 2. Uji coba | Pompa panas dipasang di sebagian rumah, dan zona serupa yang belum mendapat program dipakai sebagai pembanding | Memisahkan pengaruh program dari pengaruh lain, misalnya cuaca atau harga energi |
| 3. Pengukuran | Pemakaian pemanas listrik dibandingkan sebelum dan sesudah program, terhadap zona pembanding | Membuktikan apakah bantuan tersebut benar-benar mendorong warga beralih |
| 4. Perluasan | Program diperluas ke zona Q2 lain bila hasilnya terbukti | Menghindari anggaran besar sebelum ada bukti |

Langkah 1 menghubungkan R3 dengan R2, karena penilaian gardu adalah inti rekomendasi kedua.

Versi v5 menyebut "audit energi" pada langkah 2. Tujuan audit pada R3 tidak dijelaskan di *deck*, sehingga di sini dihapus. Bila ingin dipertahankan, tuliskan "audit energi rumah sebelum pemasangan".

## 8. Keterbatasan

- **Pola komposisi, bukan sebab-akibat** (bagian 5).
- **Angka adopsi berasal dari kawasan latih.** Kawasan uji tidak memiliki nilai aktual.
- **Data tidak memuat biaya pompa panas, biaya pembongkaran, atau harga energi**, sehingga besar bantuan belum dapat dihitung.
- **Data tidak memuat beban puncak dan kapasitas gardu**, sehingga pemeriksaan gardu tetap diperlukan.
- **Posisi Q2 bergantung pada asumsi COP.** Hanya 33 dari 47 zona Q2 yang tetap Q2 pada skenario COP per iklim.

## 9. Pertanyaan juri dan jawabannya

**Mengapa disebut pola komposisi?**
Porsi jenis pemanas di satu kawasan berjumlah 100%. Bila lebih dari 80% rumah memakai gas pipa, sisa untuk pemanas listrik paling banyak di bawah 20%. Jadi, adopsi yang rendah sebagian terjadi dengan sendirinya. Temuan ini menunjukkan di mana elektrifikasi belum terjadi, bukan mengapa.

**Apakah pemanas listrik hanya ada di kawasan tanpa gas pipa?**
Tidak. Sebanyak 82,92% rumah tangga berpemanas listrik justru tinggal di kawasan dengan gas pipa di atas 60%, karena kawasan seperti itu merupakan mayoritas. Notebook bagian 7.3 butir 4 menulis "hampir hanya", dan kalimat itu kurang tepat.

**Mengapa tidak membandingkan dengan 40,24%?**
Kelompok dengan gas pipa 40% atau kurang hanya mencakup 0,55% kawasan, sehingga perbandingannya akan berlebihan.

**Mengapa Q2, bukan Q1?**
Kenaikan listrik Q1 di atas median, sehingga kebutuhan panasnya perlu diturunkan lebih dahulu melalui renovasi (R1). Kenaikan listrik Q2 di bawah median, sehingga pompa panas dapat diuji lebih dahulu di sana.

**Mengapa bukan Q4?**
Intensitas gas Q4 di bawah median, sehingga emisi yang dapat diturunkan lebih kecil. Q4 bukan prioritas awal.

**"Kenaikan listrik di bawah median" itu berapa?**
Rata-rata Q2 masih sekitar +90%. Angka ini tidak kecil, hanya lebih rendah daripada setengah zona lainnya. Karena itu, uji coba didahului pemeriksaan gardu.

**Mengapa REG_033?**
REG_033 adalah salah satu dari 33 zona Q2 yang posisinya stabil pada kedua skenario COP. Porsi gas pipanya tinggi (83,21%) dan pemakaian pemanas listriknya rendah (4,99%).

**Mengapa R² model adopsi setinggi 0,99?**
Porsi jenis pemanas saling melengkapi hingga 100%, sehingga adopsi hampir dapat dihitung dari porsi pemanas lain. R² yang tinggi tidak berarti model memahami perilaku warga.

## 10. Usulan teks slide 11

**Judul**
> Rekomendasi 3: uji coba pompa panas di zona Q2, dimulai dari kawasan bergas

**TEMUAN**
> Di kawasan yang lebih dari 80% rumahnya memakai gas pipa, hanya 2,73% rumah yang memakai pemanas listrik, sedangkan di kawasan dengan gas pipa 60–80% angkanya 9,46%. SHAP menunjukkan pola yang sama.

**INTERPRETASI**
> Kawasan bergas, yang mencakup 70% kawasan, tertinggal dalam elektrifikasi. Pola komposisi, bukan sebab-akibat.

**REKOMENDASI**
> Pemerintah kota memulai uji coba pompa panas di zona Q2, termasuk bantuan pemasangan dan pembongkaran pemanas gas lama.
>
> Zona Q2 dipilih karena konsumsi gasnya tinggi, sedangkan kenaikan listriknya relatif rendah.
>
> Contohnya REG_033: 83,21% rumah di zona ini memakai gas pipa, tetapi baru 4,99% yang memakai pemanas listrik.

**LANGKAH**
> 1. Periksa kapasitas gardu di zona uji coba
> 2. Uji coba pompa panas, dengan zona serupa sebagai pembanding
> 3. Ukur kenaikan pemakaian pemanas listrik
> 4. Perluas ke zona Q2 lain bila hasilnya terbukti

**Pelaksana:** pemerintah kota dan penyedia pompa panas

**Baris kaki (12 pt)**
> Sumber: notebook bagian 7.1 dan 7.3, In [33] sampai [35]; pemakaian pemanas listrik per kelompok dan REG_033: skrip cek terpisah

**Anotasi grafik SHAP** (ganti "gas jaringan" dengan "gas pipa")
> Sumbu mendatar: porsi pemanas gas pipa di kawasan (%)

## 11. Usulan naskah

Sekitar 55 detik.

> "Rekomendasi ketiga menyangkut pemakaian pemanas listrik. Di kawasan yang lebih dari 80 persen rumahnya memakai gas pipa, hanya 2,73 persen rumah yang memakai pemanas listrik. Di kawasan dengan gas pipa 60 sampai 80 persen, angkanya 9,46 persen. Kawasan bergas seperti ini mencakup 70 persen kawasan, sehingga sebagian besar kota tertinggal dalam elektrifikasi. Model menunjukkan pola yang sama, dan ini pola komposisi, bukan sebab-akibat. Karena itu, kami merekomendasikan uji coba pompa panas di zona Q2, termasuk bantuan pembongkaran pemanas gas lama. Q2 dipilih karena konsumsi gasnya tinggi, sementara kenaikan listriknya masih di bawah median. Salah satu contohnya zona REG_033. Uji coba dimulai dengan pemeriksaan gardu, dibandingkan dengan zona pembanding, dan diperluas bila hasilnya terbukti."
