# Outline Presentasi Final (draf)

Format: 10 menit paparan + 10 menit tanya jawab (Zoom, 30 Oktober). Sembilan slide, sekitar satu menit per slide.
Figur ada di `reports/figures/` (300 dpi). Angka bertanda [[...]] diisi setelah freeze final.

## Alur slide

### Slide 1. Judul dan hasil utama (30 detik)
- Judul: "Memahami Struktur Data Dulu, Baru Memodelkan: Prediksi Trilema Transisi Energi Perkotaan"
- Tim [[TEAM_NAME]]. Public LB [[LB_FINAL]], CV nested [[CV_FINAL]], pipeline deterministik (sha256 sama di dua run bersih dan notebook).
- Catatan pembicara: buka dengan satu kalimat inti. "Skor kami datang dari tiga hal: identitas fisik di data, validasi yang meniru cara test dibuat, dan ensemble yang dinilai secara jujur."

### Slide 2. Masalah dan metrik (1 menit)
- Lima target memotret tiga sisi trilema: skala konsumsi (gas, listrik), intensitas termal (EUI), kualitas stok bangunan (rasio bangunan bersih), dan penetrasi teknologi (adopsi pemanas listrik).
- Metrik: rata-rata RMSE per kolom dibagi simpangan baku beku. Setiap kolom berbobot sama, jadi setiap target boleh punya pipeline sendiri.
- Konsekuensi: EUI menyumbang sekitar 52% galat kami, listrik 25%, gas 20%, adopsi 3%. Fokus kerja mengikuti porsi itu.
- Figur: `fig1_targets.png`.

### Slide 3. Memahami data sebelum memodelkan (1,5 menit)
- Identitas eksak:
  - EUI x `bld_dim_06` = gas sampai presisi mesin. Jadi `bld_dim_06` adalah luas lantai dan satu model gas sekaligus memberi EUI.
  - Rasio bangunan bersih = round(100 x `bld_dim_02` / `bld_dim_01`, 2) di 100% baris. Target ini dihitung, bukan diprediksi.
  - Adopsi pemanas listrik konstan dalam subzona (unit sensus). Untuk 92% baris test nilainya diketahui dari train.
- Forensik split: porsi test di subzona baru 8,14% (harapan split acak 8,17%), adversarial validation AUC 0,50. Test adalah sampel acak per baris.
- Implikasi: KFold acak per baris meniru test, dan statistik wilayah (target encoding) sah dipakai asalkan dihitung out-of-fold.
- Catatan pembicara: tekankan bahwa dua dari lima target "selesai" sebelum model apa pun dilatih.

### Slide 4. Berapa yang sebenarnya bisa dijelaskan? (1 menit)
- Varians residual log mengikuti a + b/M (M = jumlah meter). Pola ini khas rata-rata konsumsi rumah tangga.
- Pada blend final, 73% varians residual gas dan 77% listrik adalah noise rata-rata rumah tangga. Lantai skor dari fitur saja sekitar 0,21.
- Implikasi: model yang lebih besar tidak banyak membantu. Yang membantu adalah mengurangi varians (ensemble, rata-rata fold) dan bobot sampel yang sesuai struktur noise.
- Figur: `fig2_noise_floor.png`.

### Slide 5. Formulasi target dan fitur (1,5 menit)
- Target sebagai laju terhadap eksposur: gas per meter, per luas lantai, per luas selubung, per sertifikat energi, serta log-nya. Bobot sampel dibuat agar loss setara MSE skala asli.
- Objektif Poisson (GLM laju dengan eksposur) untuk GBM: model tunggal membaik 0,0025 sampai 0,0039.
- Fitur bertarget, semuanya cross-fitted: target encoding subzona, zona, jendela nomor subzona yang berurutan secara spasial, sel grid iklim di luar subzona sendiri, dan kNN-target (rata-rata target kawasan dengan stok bangunan paling mirip).
- Figur: `fig5_gains.png` (panel atas).

### Slide 6. Ensemble dan koreksi residual (1,5 menit)
- [[N_MEMBERS]] anggota: XGBoost, LightGBM, MLP, MLP dengan embedding periodik (PLR), TabM-mini, MLP multi-task.
- Blend NNLS per kolom, dinilai secara nested. Blend memperbaiki anggota terbaik: EUI 0,0088, gas 0,0044, listrik 0,0023.
- Koreksi residual: residual blend masih berkorelasi antar kawasan yang mirip dan berdekatan. Prediksi digeser oleh rata-rata residual (di-shrink) dari zona, sel iklim, jendela subzona, dan k tetangga paling mirip. Bobotnya di-fit di holdout internal 70/30, bukan di OOF penuh yang bias.
- Figur: `fig4_blend_weights.png`, `fig5_gains.png` (panel bawah).

### Slide 7. Validasi yang bisa dipercaya dan reproducibility (1 menit)
- Sepuluh submisi: public LB = CV + 0,00052 (sd 0,00004). Urutan kandidat di CV selalu sama dengan di LB.
- Seed dikunci, tanpa batas waktu, versi paket dipin. XGBoost GPU, LightGBM, dan MLP terbukti identik bit-per-bit. CatBoost GPU tidak deterministik, jadi dikeluarkan (biaya CV sekitar 0,00001).
- Dua run bersih dan eksekusi notebook memberi sha256 [[SHA]].
- Figur: `fig3_cv_lb.png`.

### Slide 8. Insight kebijakan (1,5 menit)
- [[Isi dari SHAP per keluarga fitur, figur shap_t3.png, shap_t2.png, shap_t5.png]]
- Dilema 1, skala vs intensitas termal: gas total digerakkan ukuran kawasan (luas lantai, jumlah meter), sedangkan EUI digerakkan [[keluarga utama EUI]].
- Dilema 2, beban jaringan listrik: [[pendorong listrik per meter dan kaitannya dengan adopsi pemanas listrik]].
- Dilema 3, kesenjangan adopsi: [[pendorong adopsi, tenure dan sosio-demografi]].

### Slide 9. Kesimpulan dan pengembangan (30 detik)
- Tiga pelajaran: (1) cari struktur fisik dulu, (2) bangun validasi yang meniru test lalu buktikan dengan LB, (3) nilai ensemble dan post-processing secara jujur.
- Keterbatasan: sekitar tiga perempat varians residual adalah noise rata-rata rumah tangga. Lompatan berikutnya butuh data per rumah tangga, bukan model yang lebih besar.

## Pertanyaan juri yang mungkin muncul

1. **Mengapa KFold acak, bukan GroupKFold per zona seperti saran panitia?**
   Kami menguji cara test dibuat. Porsi test di subzona baru 8,14% sementara harapan split acak 8,17%, proporsi test per zona tersebar binomial, dan adversarial validation memberi AUC 0,50. Test adalah sampel acak per baris, jadi KFold acak yang meniru kondisi itu. GroupKFold per zona tetap kami hitung sebagai uji ketahanan, dan selisih CV dengan LB yang stabil (sd 0,00004) mengonfirmasi pilihan ini.

2. **Bagaimana mencegah kebocoran pada target encoding dan fitur kNN-target?**
   Semua statistik bertarget dihitung out-of-fold, nested di dalam fold training. Kami mengujinya dengan mengacak label baris validasi lalu memastikan fitur validasi tidak berubah. Bobot koreksi residual di-fit di holdout internal, tidak di OOF penuh.

3. **Ensemble sebesar itu, apakah tidak overfit ke CV?**
   Bobot NNLS non-negatif, dan skor blend selalu dinilai nested: bobot di-fit di fold lain lalu dinilai di fold yang tidak ikut fitting. Bukti di luar sampel adalah LB: sepuluh submisi mengikuti CV dengan selisih tetap.

4. **Bagaimana reproducibility dijamin?**
   Seed dikunci di semua library, tidak ada `time_limit`, urutan kolom dari list yang di-sort, versi paket dipin. Kami menguji determinisme per keluarga model dan membuang CatBoost GPU yang tidak identik antar run. Dua run bersih dan eksekusi notebook menghasilkan sha256 yang sama.

5. **Mengapa EUI paling sulit?**
   EUI hanya berkorelasi lemah dengan fitur tunggal (maksimum 0,18) dan noisenya membesar di kawasan kecil (varians sebanding 1/luas). EUI menyumbang sekitar setengah galat kami. Kami memakai bobot invers-varians dan banyak formulasi rasio untuk menekannya.

6. **Apa arti pola a + b/M?**
   Gas dan listrik kawasan adalah jumlah konsumsi rumah tangga. Bila konsumsi per rumah tangga bervariasi acak, varians relatif total turun sebanding 1/M. Bagian b/M tidak bisa dijelaskan fitur apa pun, sedangkan bagian a adalah galat model. Pada blend final a sekitar 0,009 sampai 0,012, sekitar seperempat varians residual.

7. **Mengapa objektif Poisson?**
   Laju per eksposur dengan varians sebanding rata-rata adalah GLM klasik untuk data konsumsi. Pada model tunggal GBM gain-nya besar (EUI 0,0039). Di blend gain-nya kecil karena ensemble sudah menyerap efek yang sama, dan kami melaporkannya apa adanya.

8. **Apakah memakai data eksternal?**
   Tidak. Semua nilai prediksi berasal dari data resmi dan kode open-source yang dijalankan lokal. Identitas eksak ditemukan dari EDA pada data yang diberikan.

9. **Bagaimana dengan subzona yang tidak muncul di train (sekitar 8% baris test)?**
   Encoding jatuh ke tingkat zona, jendela subzona, dan sel iklim. Untuk adopsi, jumlah rumah tangga N direkonstruksi dari pecahan `hvac_dist`, lalu cacah kategori yang hilang dimodelkan secara binomial. Blend dan koreksi terpisah untuk baris ini sudah kami uji dan tidak menambah skor.

10. **Mengapa koreksi residual tidak bocor?**
    Saat fitting bobot, baris holdout berperan sebagai baris test: tetangganya hanya diambil dari bagian 70%, dengan residual OOF yang tidak pernah melihat label holdout. Saat inferensi, tetangga diambil dari residual OOF train. Bobot tidak pernah di-fit di OOF penuh karena di sana residual tetangga sudah tercemar lewat target encoding.

11. **Insight kebijakan apa yang didapat?**
    [[Diisi dari SHAP: tiga dilema.]]

12. **Berapa lama pipeline berjalan dan butuh hardware apa?**
    [[jam]] di satu RTX A4000 16 GB dan 16 core CPU. Notebook kedua membuat ulang submisi yang sama dari model tersimpan dalam waktu jauh lebih singkat.

13. **Apa saja yang tidak berhasil?**
    Stacking level-2 dengan LightGBM, blend per kelompok ukuran, blend khusus subzona baru, loss Poisson pada MLP, CatBoost di pipeline final (tidak deterministik), dan pencarian pola pada nilai gas "kembar". Semua tercatat di registry beserta skornya.

14. **Apa yang akan dilakukan dengan waktu lebih?**
    Kuantifikasi ketidakpastian per kawasan (interval prediksi sebanding a + b/M) agar perencana jaringan tahu kawasan mana yang prediksinya kurang pasti, dan model adopsi yang memakai informasi tingkat rumah tangga bila tersedia.
