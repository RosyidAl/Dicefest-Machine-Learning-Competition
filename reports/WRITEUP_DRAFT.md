# Memprediksi Trilema Transisi Energi Perkotaan dengan Struktur Fisik Data dan Ensemble yang Reproducible

DICE FEST 2026, Machine Learning. Tim: [[TEAM_NAME]]

## Abstrak
Kami memprediksi lima indikator energi untuk lebih dari 600 ribu kawasan permukiman: konsumsi gas, konsumsi listrik, intensitas gas per meter persegi (EUI), rasio bangunan bersih, dan adopsi pemanas listrik.

Kunci pendekatan kami adalah memahami struktur data sebelum memilih model. Ada tiga hal yang kami temukan:
- Dua target bisa dihitung secara eksak dari fitur.
- Gas dan EUI terikat identitas fisik, gas = EUI x luas lantai.
- Data test diambil acak per baris, sehingga informasi tingkat wilayah dapat dipakai secara sah.

Di atas fondasi itu kami membangun:
- [[N_MEMBERS]] model cross-validated (XGBoost, LightGBM, dan MLP dengan embedding periodik);
- blend non-negatif per kolom;
- koreksi residual berbasis kemiripan stok bangunan di sekitar setiap kawasan.

Skor validasi silang jujur (nested) kami [[CV_FINAL]] dan public leaderboard [[LB_FINAL]]. Seluruh pipeline deterministik: dua run dari kondisi bersih dan satu eksekusi notebook menghasilkan file submisi dengan sha256 yang sama.

## 1. Pendahuluan dan pemahaman masalah
Transisi energi di kota menghadapi tiga tarikan sekaligus:
- menurunkan emisi pemanasan berbasis gas;
- menjaga jaringan listrik tetap sanggup ketika pemanas beralih ke listrik;
- memastikan manfaatnya tidak hanya dinikmati kelompok tertentu.

Kelima target kompetisi memotret ketiga sisi itu:
- skala konsumsi (gas dan listrik total);
- intensitas termal (EUI);
- kualitas stok bangunan (rasio bangunan bersih);
- penetrasi teknologi baru (adopsi pemanas listrik).

Metrik lomba adalah rata-rata RMSE tiap target dibagi simpangan baku bekunya. Karena dihitung per kolom, setiap target boleh punya pipeline sendiri, dan meminimalkan MSE tiap kolom sama dengan meminimalkan metrik. Prediksi konstan memberi skor sekitar 1,0 per target.

## 2. Data dan eksplorasi
Train berisi 429.303 baris dan test 183.987 baris. Ada 201 fitur anonim dalam lima keluarga (dimensi bangunan, era konstruksi, bauran pemanas, iklim, sosio-demografi) ditambah dua ID spasial: 330 zona dan sekitar 150 ribu subzona.

Temuan yang paling menentukan arah pemodelan:
1. **Identitas eksak.**
   - EUI dikali `bld_dim_06` sama dengan gas sampai presisi mesin (selisih relatif maksimum 7e-16). Jadi `bld_dim_06` adalah luas lantai, dan satu model gas sekaligus memberi EUI.
   - Rasio bangunan bersih sama persis dengan round(100 x `bld_dim_02`/`bld_dim_01`, 2) di 100% baris.
2. **Adopsi pemanas listrik konstan di dalam subzona** (subzona adalah unit sensus). Bauran pemanas `hvac_dist_*` adalah pecahan cacah per jumlah rumah tangga N, sehingga N bisa direkonstruksi eksak sebagai penyebut terkecil yang membuat semua pecahan bulat. Adopsi sama dengan sisa kategori pemanas dikurangi satu kategori yang hilang.
3. **Target makro didominasi ukuran kawasan.** Korelasi Spearman gas dengan luas lantai 0,92, dan listrik dengan jumlah meter 0,90. Sebaliknya, EUI hanya berkorelasi lemah dengan fitur tunggal (maksimum 0,18), dan EUI menyumbang sekitar dua pertiga skor tim mana pun.
4. **Noise label besar.** Varians residual log mengikuti pola a + b/M (M = jumlah meter), khas rata-rata konsumsi rumah tangga. Sekitar tiga perempat varians residual adalah noise yang tidak bisa dijelaskan fitur, sehingga lantai skor dari fitur saja sekitar 0,21.

## 3. Strategi validasi
Sebelum memilih skema cross-validation, kami memeriksa bagaimana data test dibuat:
- Rasio train/total tepat 0,700000.
- Proporsi test per zona tersebar persis binomial (std z-score 0,989).
- Porsi baris test di subzona yang tidak muncul di train adalah 8,14%, hampir sama dengan harapan split acak per baris (8,17%).
- Adversarial validation (LightGBM) memberi AUC 0,50, artinya train dan test tidak bisa dibedakan.

Kesimpulannya, test adalah sampel acak 30% per baris. Karena itu CV utama memakai KFold acak per baris (5 dan 10 fold, seed 42), dan statistik tingkat wilayah (target encoding subzona, zona, jendela spasial) sah dipakai selama dihitung out-of-fold.

CV sekunder dengan fold per zona (saran panitia) kami laporkan sebagai uji ketahanan. Selisih CV dengan public leaderboard stabil di [[N_SUB]] submisi, dan urutan kandidat selalu sama.

## 4. Rekayasa target dan fitur
**Formulasi target.** Karena gas = EUI x luas, semua model keluarga gas memprediksi kuantitas per unit lalu dikalikan kembali. Pilihan unitnya beragam:
- per meter;
- per luas lantai;
- per luas selubung (GEA);
- per jumlah sertifikat energi;
- logaritmanya.

Bobot sampel diatur agar loss setara dengan MSE pada skala asli (misalnya bobot M²/A untuk gas per meter). Keragaman formulasi ini ternyata sumber keragaman ensemble terbesar.

**Fitur.**
- Rasio fisik (luas per meter, faktor bentuk).
- Agregat transductive per subzona dan zona (jumlah baris, total luas, porsi baris).
- Target encoding out-of-fold untuk subzona, zona, jendela ±2 dan ±10 nomor subzona (yang berurutan secara spasial), serta sel grid iklim di luar subzona sendiri.
- **Fitur kNN-target (FB07):** rata-rata target dari k kawasan dengan stok bangunan paling mirip dalam jendela spasial. Pada XGBoost tunggal fitur ini menurunkan NRMSE gas 0,0011 dan EUI 0,0022, membaik di kelima fold.

Semua fitur bertarget dihitung dengan cross-fitting dan lulus uji kebocoran: label baris validasi diacak, lalu fitur validasi diperiksa tetap identik.

## 5. Pemodelan
[[Tabel keluarga model x target, jumlah anggota, waktu.]]
- XGBoost (CUDA, deterministik), LightGBM (CPU, deterministic=True), dan MLP PyTorch dengan embedding periodik (PLR), TabM-mini, serta MLP multi-task yang mempelajari gas dan listrik bersamaan.
- MLP dilatih dengan jumlah epoch tetap (one-cycle), tanpa memilih epoch dari fold validasi, sehingga OOF tetap jujur.
- CatBoost GPU diuji dan kuat, tetapi tidak deterministik, sehingga dikeluarkan dari pipeline final dengan biaya CV sekitar 0,00001.

## 6. Ensemble dan koreksi residual
- Blend memakai non-negative least squares per kolom yang dinilai, dengan bobot dari prediksi OOF. Kami membandingkannya dengan hill climbing, rata-rata top-N, Ridge positif, stacking LightGBM, dan MLP level-2; [[hasil EN]].
- Koreksi residual: residual OOF blend masih berkorelasi spasial. Prediksi test digeser oleh kombinasi linear rata-rata residual (yang di-shrink) dari zona yang sama, sel iklim yang sama, jendela ±30 subzona, dan k tetangga dengan stok bangunan paling mirip. Bobot kombinasinya di-fit pada holdout internal 70/30, tidak pada OOF penuh yang bias. [[gain holdout]].
- Adopsi: untuk subzona yang ada di train, nilainya disalin. Untuk 8% baris di subzona baru, cacah kategori hilang dimodelkan secara binomial dengan LightGBM.
- 77 baris test identik dengan baris train, dan targetnya disalin.

## 7. Hasil dan ablation
[[Tabel: baseline B0-B6 di CV primer/sekunder, transform T0-T8, ablation keluarga fitur, model zoo, ensemble, CV vs LB.]]

## 8. Interpretasi kebijakan
[[SHAP per keluarga fitur per target dan tiga dilema: skala vs intensitas termal (gas vs EUI), beban jaringan (listrik dan adopsi), kesenjangan adopsi dan tenure.]]

## 9. Reproducibility
- Seed dikunci di semua tahap, tidak ada batas waktu, dan versi paket dipin.
- Notebook mandiri menyematkan seluruh kode.
- Dua run bersih dan satu eksekusi notebook headless memberi sha256 [[SHA]].
- Notebook kedua membuat ulang submisi yang sama dari model tersimpan dalam waktu jauh lebih singkat.

## 10. Keterbatasan dan pengembangan
- Sekitar 75% varians residual gas dan listrik berasal dari rata-rata rumah tangga yang tidak teramati. Perbaikan berikutnya butuh data per rumah tangga, bukan model yang lebih besar.
- Pipeline penuh butuh [[jam]] di satu GPU.

## 11. Kesimpulan
[[ringkasan]]
