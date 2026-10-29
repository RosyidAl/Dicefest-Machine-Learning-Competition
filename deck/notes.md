# Naskah Pembicara: Pitch Deck BISMILLAH (8 menit)

Sumber angka: `final/deliverables/WRITEUP_BISMILLAH_B_bab_tetap.pdf` dan `reports/deck_facts_extra.md`.
Slide 1 sampai 11 dibawakan dalam 8 menit; slide 12 sampai 18 adalah lampiran untuk tanya jawab.
Pembagian waktu mengikuti bobot juri: kebijakan 3 menit 20 detik (slide 8 sampai 10), fitur dan pra-pemrosesan
2 menit 30 detik (slide 3 sampai 6), arsitektur dan kode 1 menit 20 detik (slide 7 dan 11), pembuka 50 detik.

## C. Isi per slide

### Slide 1. Judul (20 detik)
- **Judul:** Dari Struktur Data ke Peta Kebijakan
- **Catatan pembicara:** Kami tim BISMILLAH. Kami memprediksi lima besaran energi untuk 613 ribu kawasan permukiman, lalu menerjemahkan model yang sama menjadi peta prioritas untuk 330 zona. Skor private kami 0,24844 di peringkat 8, dan skor publik 0,24913.

### Slide 2. Lima target, tiga tuntutan (30 detik)
- **Judul:** Lima target mencerminkan tiga tuntutan transisi energi
- **Catatan pembicara:** Transisi dari gas ke pemanas listrik menghadapkan kota pada tiga tuntutan: emisi, keandalan jaringan, dan keterjangkauan biaya. Kelima target mencerminkan ketiganya. Setiap target dinilai terpisah, jadi masing-masing boleh ditangani dengan cara berbeda. Pada model akhir, EUI menyumbang sekitar separuh galat, sehingga di sanalah upaya kami paling banyak tercurah.

### Slide 3. Struktur data (40 detik)
- **Judul:** Dua target terjawab oleh struktur data sebelum model dilatih
- **Catatan pembicara:** Sebelum melatih model, kami memeriksa struktur data. Hasil kali EUI dan luas lantai selalu sama dengan gas total, selisih terbesarnya 7 kali 10 pangkat min 16, jadi keduanya dimodelkan bersama. Porsi bangunan bersih cocok persis dengan satu rumus. Adopsi seragam di dalam subzona, dan 92% kawasan uji sudah diketahui nilainya. Data uji diambil acak per baris: AUC adversarial 0,4997.

### Slide 4. Skema validasi (40 detik)
- **Judul:** Skema validasi yang keliru akan membuang fitur terbaik
- **Catatan pembicara:** Untuk data spasial, pembagian per wilayah sering dianjurkan. Tetapi 92% kawasan uji berada di subzona yang ada di data latih. Pada skema per zona, rata-rata target wilayah tampak merusak: galat EUI 0,8331. Pada skema acak per baris yang meniru data uji, fitur yang sama menurunkannya menjadi 0,6699. Kami memilih perubahan dari holdout set 30%, bukan dari leaderboard publik.

### Slide 5. Fitur domain (45 detik)
- **Judul:** Fitur dibangun dari cara energi dipakai, bukan ukuran kawasan
- **Catatan pembicara:** Gas total sebuah kawasan terutama ditentukan jumlah rumah di dalamnya. Yang informatif adalah perilaku per rumah, jadi model memprediksi gas per meter atau per meter persegi, lalu dikalikan kembali. Rata-rata target wilayah dihitung out-of-fold, tanpa melihat fold yang sedang dievaluasi. Fitur kawasan termirip menurunkan galat EUI sebesar 0,0022, perbaikan terbesar dari satu kelompok fitur.

### Slide 6. Batas akurasi (25 detik)
- **Judul:** Tiga perempat galat berasal dari variasi antarrumah tangga
- **Catatan pembicara:** Jumlah penghuni dan kebiasaan tidak tercatat di data. Makin banyak rumah tangga dalam satu kawasan, makin kecil galatnya, mengikuti a ditambah b per M. Karena itu perbaikan datang dari ensemble, bukan dari model yang lebih besar.

### Slide 7. Arsitektur (50 detik)
- **Judul:** Tiga jalur untuk lima target, 41 model dengan pola galat berbeda
- **Catatan pembicara:** Ada tiga jalur. Gas dan EUI memakai satu kelompok model, listrik punya kelompok sendiri, dan dua target lain memakai rumus dan salinan. Totalnya 41 model: XGBoost, LightGBM, MLP, dan NCA. Bobot penggabungan berubah menurut luas kawasan, jumlah meter, dan besar prediksi. Setelah itu ada koreksi galat wilayah. Semua dilatih ulang dari awal dan menghasilkan berkas submisi yang identik byte.

### Slide 8. Pendorong konsumsi (70 detik)
- **Judul:** Ukuran rumah, periode bangunan, dan iklim menggerakkan konsumsi
- **Catatan pembicara:** Model penjelas hanya memakai karakteristik kawasan yang terukur, dan menjelaskan 75% variasi gas per meter. SHAP menunjukkan empat pola. Rumah kecil memakai gas per meter persegi sekitar 65% lebih banyak, sedangkan rumah besar memakai listrik per rumah sekitar 75% lebih banyak, jadi program efisiensi dan program emisi menyasar kawasan berbeda. Periode bld_era_15 membuat EUI sekitar 7% lebih tinggi, dan iklim menggesernya 15%. Adopsi terkonsentrasi di kawasan tanpa jaringan gas; sosiodemografi hanya 15%. Perlu dicatat, SHAP menunjukkan asosiasi, bukan sebab-akibat.

### Slide 9. Peta trilema (70 detik)
- **Judul:** Setiap zona membutuhkan urutan program yang berbeda
- **Catatan pembicara:** Kami meringkas 613 ribu kawasan menjadi 330 zona dengan dua ukuran. Potensi retrofit totalnya 7,4 TWh, 5,2% dari konsumsi gas. Elektrifikasi penuh menaikkan listrik tahunan zona rata-rata 95%, dengan asumsi 80% gas untuk pemanasan dan COP 3. Median membagi zona menjadi empat kuadran. Q1 memuat 118 zona dan 47% gas: retrofit dulu. Q2 kandidat heat pump. Q3 perlu penguatan gardu lebih dulu. Q4 cukup program standar. Peringkat zona tidak berubah oleh asumsi.

### Slide 10. Rekomendasi (60 detik)
- **Judul:** Lima langkah untuk pemerintah kota
- **Catatan pembicara:** Pertama, urutkan program per kuadran; kandidat retrofit pertama adalah REG_273, REG_270, dan REG_045. Kedua, Q2 menjadi kandidat heat pump, dengan kesiapan gardu dikonfirmasi lebih dulu. Ketiga, libatkan operator jaringan sejak awal, misalnya di REG_290 yang naik 121%. Keempat, beri insentif untuk kawasan berjaringan gas. Kelima, pisahkan target efisiensi dan target emisi. Keterbatasannya: SHAP adalah asosiasi, skenario listrik adalah energi tahunan, dan ambang kuadran berupa median.

### Slide 11. Simpulan (30 detik)
- **Judul:** Memahami data lebih berharga daripada memperbesar model
- **Catatan pembicara:** Waktu untuk memahami struktur data jauh lebih bernilai daripada mencari model yang lebih besar. Setiap gagasan divalidasi sebelum dipakai, dan setiap hasil bisa direproduksi: seed dikunci, tanpa batas waktu, dan notebook kedua mereproduksi submisi dari model tersimpan. Terima kasih.

### Slide cadangan (untuk tanya jawab)
- **A1. Tabel 1 validasi:** galat LightGBM dengan dan tanpa rata-rata target wilayah pada dua skema.
- **A2. Komposisi ensemble:** jumlah model, galat terbaik, dan bobot per kelompok.
- **A3. Sensitivitas skenario listrik:** porsi pemanasan 70/80/90% dan COP 2,5/3/3,5.
- **A4. Adopsi subzona baru:** N, K, q; RMSE 1,03 poin persentase.
- **A5. Audit akhir:** kawasan 2 sampai 4 meter, batas gas 999.765 kWh, pendekatan yang gagal.
- **A6. Tabel 3 SHAP:** porsi kelompok fitur dan zona kandidat.
- **A7. Data dan metrik:** ukuran data, sigma, perbaikan ensemble.

## D. Daftar periksa angka

| Angka | Nilai |
|---|---|
| Skor *leaderboard* private / publik | 0,24844 (peringkat 8) / 0,24913 |
| Data | 429 ribu kawasan latih, 184 ribu kawasan uji, 201 kolom fitur, 330 zona |
| Identitas gas = EUI × luas | selisih relatif terbesar 7 × 10⁻¹⁶ |
| Kawasan uji dengan adopsi diketahui | 92% |
| AUC validasi adversarial | 0,4997 |
| Galat EUI per zona vs acak (dengan rata-rata wilayah) | 0,8331 vs 0,6699 |
| Perbaikan fitur kawasan termirip | EUI 0,0022 |
| Jumlah model | 41 (28 gas/EUI, 13 listrik) |
| R² model penjelas | 75% gas per meter, 56% EUI, 49% listrik per meter |
| Pendorong | 65%, 75%, 7%, 15%, 15% |
| Potensi *retrofit* | 7,4 TWh dari 141,4 TWh (5,2%) |
| Kenaikan konsumsi listrik tahunan | rata-rata 95%, antarzona 68% sampai 121% |
| Kuadran | Q1 118 zona, 47% gas, +102%; Q2 47, 12%, +90%; Q3 47, 15%, +99%; Q4 118, 27%, +87% |
