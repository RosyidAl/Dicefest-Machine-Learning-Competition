## First Principles Analysis: Trilema Transisi Energi Perkotaan

Analisis ini membongkar permasalahan inti *case* DICE FEST 2026 dari kebenaran dasar (*ground truths*), bukan dari analogi atau kebiasaan industri. Tujuannya: memahami **mengapa** masalah ini sulit, **apa** yang benar-benar harus dijawab, dan **bagaimana** pendekatan tim BISMILLAH menjawabnya.

---

### 1. Esensi Masalah

**Masalah inti (satu kalimat):**
Sebuah kota ingin beralih dari gas ke pemanas listrik, tetapi tidak tahu **di mana mulai** dan **dalam urutan apa**, karena setiap kawasan permukiman punya profil konsumsi energi yang berbeda, dan bertindak tanpa urutan yang tepat bisa memicu tiga kegagalan sekaligus: emisi tidak turun, jaringan listrik kolaps, atau biaya melonjak.

**Kriteria sukses yang sebenarnya:**
1. Prediksi lima besaran energi per kawasan yang cukup akurat untuk membedakan kawasan satu dari lainnya (bukan sekadar skor rendah di *leaderboard*).
2. Peta prioritas 330 zona yang bisa ditindaklanjuti: zona mana yang di-*retrofit* dulu, mana yang butuh penguatan gardu, mana yang siap *heat pump*.
3. Rekomendasi yang tahan terhadap asumsi: urutan prioritas tidak berubah meskipun asumsi skenario bergeser.

**Mengapa ini bukan sekadar regresi biasa:**
Kompetisi menyajikannya sebagai "multi-output regression", tetapi esensi masalahnya adalah **alokasi sumber daya di bawah ketidakpastian**. Model prediksi hanyalah alat perantara; nilai sesungguhnya terletak pada keputusan kebijakan yang dibangun di atasnya.

---

### 2. Asumsi yang Ditantang

| Asumsi | Tantangan | Putusan |
|--------|-----------|---------|
| "Lima target harus dimodelkan secara terpisah dengan lima model berbeda" | Gas total = EUI x luas lantai, selisih 7e-16. Porsi bangunan bersih = rumus pasti dari dua kolom. Adopsi konstan per subzona, 92% data uji sudah diketahui. | **Modifikasi.** Hanya gas/EUI dan listrik yang butuh model ML penuh. Dua target lain dijawab oleh struktur data, satu lagi oleh salinan. |
| "Validasi spasial (per wilayah) lebih realistis untuk data geografi" | Data uji diambil acak per baris (AUC adversarial 0,4997), bukan per zona. 92% kawasan uji sudah ada subzonanya di data latih. Pada skema per zona, rata-rata target wilayah tampak merusak (EUI 0,833 dengan fitur itu vs 0,692 tanpa), padahal pada skema acak yang meniru data uji fitur yang sama menurunkan EUI 0,0022. | **Buang.** CV acak per baris meniru mekanisme split yang sebenarnya. Validasi per zona dilaporkan sebagai pelengkap, bukan pengarah keputusan. |
| "Model yang lebih besar dan lebih banyak akan selalu lebih baik" | Tiga perempat galat berasal dari variasi antarpelanggan (jumlah penghuni, kebiasaan) yang tidak tercatat di data. Kurva galat mengikuti a + b/M (M = jumlah meter). Di luar titik tertentu, menambah model tidak menurunkan galat; hanya ensemble 41 model dengan bobot adaptif yang memberi perbaikan marginal. | **Modifikasi.** Perbaikan datang dari pemahaman data (fitur fisik, kawasan mirip) dan ensemble yang cerdas, bukan dari model individual yang lebih besar. |
| "Fitur harus dibuat dari ukuran kawasan" | Yang informatif adalah perilaku per rumah, bukan ukuran kawasan. Model memprediksi gas per meter atau per meter persegi, lalu dikalikan kembali. `area_per_meter` (ukuran rata-rata satu rumah) adalah fitur SHAP terbesar. | **Buang.** Fitur dibangun dari cara energi dipakai per unit, bukan dari skala kawasan. |
| "SHAP = kausalitas = rekomendasi kebijakan langsung" | SHAP menunjukkan asosiasi, bukan sebab-akibat. Kontribusi komposisi pemanas pada adopsi (81%) sebagian besar adalah identitas akuntansi. | **Modifikasi.** SHAP dipakai sebagai penunjuk arah investigasi, bukan sebagai bukti kausal. Rekomendasi diturunkan dari logika fisik yang didukung SHAP, bukan dari SHAP semata. |
| "Skor *leaderboard* adalah satu-satunya ukuran keberhasilan" | Juri final menilai: variabel berpengaruh dan kebijakan (35 poin), fitur dan pra-pemrosesan (25 poin), presentasi (25 poin), kode (15 poin). Skor private LB hanya 95% penentu finalis, bukan penentu juara. | **Buang.** Memahami data dan menurunkan rekomendasi kebijakan lebih bernilai daripada memeras 0,001 dari skor. |

---

### 3. Kebenaran Dasar (*Ground Truths*)

Berikut fakta yang tidak bisa direduksi lebih jauh dan akan rusak jika dilanggar:

1. **Identitas definisi gas dan EUI.** EUI didefinisikan sebagai gas dibagi luas lantai (`bld_dim_06`), sehingga gas = EUI x luas lantai. Selisih relatifnya 6,7e-16 (presisi *floating-point*). Implikasi: memodelkan EUI otomatis memodelkan gas total.

2. **Variabilitas antarpelanggan tak teramati.** Data hanya merekam agregat kawasan, bukan perilaku setiap rumah tangga. Jumlah penghuni, jam pemakaian, dan preferensi suhu tidak ada. Ini menetapkan **batas bawah galat** yang tidak bisa ditembus oleh model apapun. Galat mengikuti a + b/M: makin sedikit rumah tangga di kawasan, makin besar ketidakpastian.

3. **Hierarki spasial menentukan informasi.** Kawasan dalam subzona yang sama berbagi infrastruktur: jaringan gas, gardu listrik, stok bangunan serupa. 99% baris berurutan menurut nomor subzona berada di zona yang sama. Rata-rata target wilayah (dihitung OOF) menurunkan galat EUI 0,0022 pada CV acak.

4. **Karakteristik bangunan berasosiasi dengan konsumsi** (hasil SHAP, bukan bukti sebab-akibat).
   - Rumah kecil memakai gas per m2 sekitar 65% lebih banyak.
   - Rumah besar memakai listrik per rumah sekitar 75% lebih banyak.
   - Periode bangunan menentukan kualitas insulasi: era `bld_era_15` menaikkan EUI sekitar 7%.
   - Iklim (`env_climate_05`, beban *heating degree-days*) menggeser konsumsi sekitar 15%.

5. **Trilema adalah *trade-off* nyata, bukan retorika.**
   - **Emisi:** mengurangi gas membutuhkan *retrofit* atau elektrifikasi.
   - **Keandalan jaringan:** elektrifikasi penuh menaikkan konsumsi listrik zona rata-rata 95% (68% sampai 121% antarzona). Gardu yang tidak diperkuat akan *overload*.
   - **Keterjangkauan:** *retrofit* dan *heat pump* membutuhkan investasi awal tinggi; kawasan miskin paling butuh tetapi paling sulit membayar.
   - Ketiga tuntutan ini saling berkompetisi: mengejar satu tanpa mempertimbangkan dua lainnya akan gagal.

6. **Adopsi pemanas listrik hampir sepenuhnya ditentukan oleh ketersediaan gas.** `hvac_rest` (sisa porsi di luar pemanas yang tercatat) berkorelasi 0,988 dengan adopsi. Di kawasan tanpa jaringan gas, adopsi listrik sudah tinggi. Sosiodemografi hanya menyumbang 15%.

---

### 4. Rantai Penalaran

```
Kebenaran Dasar                 Langkah 1                        Langkah 2                         Solusi
─────────────────────           ──────────────────────            ──────────────────────             ──────────────────────
Gas = EUI x Luas           →   Cukup modelkan EUI,              →   Satu pipeline gas/EUI          →   28 model gabungan
                                gas mengikuti otomatis                                                   untuk 2 target

Variabilitas tak teramati  →   Batas galat ada; jangan          →   Ensemble 41 model dengan        →   Bobot adaptif
mengikuti a + b/M               kejar model lebih besar,            bobot yang berubah menurut          per ukuran kawasan
                                kejar ensemble yang cerdas           luas, meter, besar prediksi

Hierarki spasial           →   Rata-rata target wilayah (OOF)   →   Fitur kawasan termirip          →   Koreksi galat
menentukan informasi            adalah fitur kunci                   (kNN-target OOF)                    wilayah di akhir

Fisika bangunan            →   Fitur per unit, bukan per        →   area_per_meter, gea_ratio,      →   SHAP menunjukkan
menggerakkan konsumsi           kawasan                              rasio era, beban iklim              pendorong untuk
                                                                                                         kebijakan

Trilema adalah              →  Ringkas 613 ribu kawasan          →  Dua ukuran: potensi retrofit    →   Empat kuadran:
trade-off nyata                 menjadi 330 zona                     (TWh) dan kenaikan listrik          urutan program
                                                                     (%)                                  berbeda per zona

Adopsi ~ ketersediaan gas  →   92% data uji sudah diketahui;    →   Model ML hanya untuk 8%         →   Rumus + salinan +
                                salinan cukup untuk mayoritas        subzona baru                         2 LightGBM fallback
```

---

### 5. Kesimpulan

**Permasalahan utama pada *case* ini:**

Bukan "memprediksi lima angka" -- itu hanya permukaan. Permasalahan utamanya adalah **menerjemahkan 613 ribu baris data anonim menjadi peta keputusan yang bisa ditindaklanjuti oleh pemerintah kota**, di mana setiap zona membutuhkan **urutan program yang berbeda** karena trilema emisi-keandalan-keterjangkauan bermanifestasi berbeda di setiap tempat.

Secara lebih rinci, ada tiga lapis masalah:

1. **Lapis data:** Nama kolom disamarkan, relasi antar-target tersembunyi, dan mekanisme split tidak dinyatakan. Tim yang langsung melatih model tanpa memahami struktur data akan membuang fitur terkuat (rata-rata wilayah) atau membuat validasi yang menyesatkan.

2. **Lapis prediksi:** Variabilitas antarpelanggan yang tak teramati menetapkan batas galat. Perbaikan tidak datang dari model yang lebih besar, melainkan dari (a) fitur yang mencerminkan fisika energi per unit, (b) informasi spasial dari kawasan mirip, dan (c) ensemble adaptif yang tahu bahwa akurasi model berbeda menurut ukuran kawasan.

3. **Lapis kebijakan:** Prediksi yang akurat tidak berguna tanpa kerangka keputusan. Potensi *retrofit* 7,4 TWh (5,2% dari total gas) dan kenaikan listrik 68-121% per zona harus dipetakan ke kuadran prioritas. Rekomendasi harus tahan terhadap perubahan asumsi (porsi pemanasan 70/80/90%, COP 2,5/3/3,5).

**Wawasan kunci dari analisis *first principles*:**
Memahami struktur data jauh lebih bernilai daripada memperbesar model. Dua dari lima target terjawab oleh struktur data sebelum model dilatih. Skema validasi yang keliru (per zona, bukan per baris) akan membuang fitur terbaik. Dan skor *leaderboard* hanyalah tiket masuk, bukan penentu juara: 60 dari 100 poin juri final ada di variabel berpengaruh dan kebijakan (35) serta fitur dan pra-pemrosesan (25).

**Kompromi yang diterima:**
- SHAP adalah asosiasi, bukan kausalitas. Rekomendasi kebijakan bersifat indikatif.
- Skenario listrik menggunakan energi tahunan, bukan beban puncak. Gardu bisa kolaps pada puncak meski energi tahunan masih aman.
- Ambang kuadran berupa median: pragmatis, tetapi tidak optimal secara teori keputusan.
- Model tidak bisa menembus batas galat dari variabilitas tak teramati.

---

### Checklist First Principles

- [x] Masalah dinyatakan dalam bentuk hasil (*outcome*), bukan solusi
- [x] Semua asumsi didaftar secara eksplisit
- [x] Setiap asumsi ditantang dan diputuskan
- [x] Kebenaran dasar diidentifikasi dan diverifikasi dari data
- [x] Solusi dibangun dari kebenaran dasar saja
- [x] Setiap keputusan desain bisa ditelusuri ke kebenaran dasar
- [x] Rantai penalaran didokumentasikan
