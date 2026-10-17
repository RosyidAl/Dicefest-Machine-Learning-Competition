# Memprediksi Trilema Transisi Energi Perkotaan dengan Struktur Fisik Data dan Ensemble yang Reproducible

DICE FEST 2026, Machine Learning. Tim: [[TEAM_NAME]]

## Abstrak
Kami memprediksi lima indikator energi untuk lebih dari 600 ribu kawasan permukiman: konsumsi gas, konsumsi listrik, intensitas gas per meter persegi (EUI), rasio bangunan bersih, dan adopsi pemanas listrik.

Kunci pendekatan kami adalah memahami struktur data sebelum memilih model. Ada tiga hal yang kami temukan:
- Dua target bisa dihitung secara eksak dari fitur.
- Gas dan EUI terikat identitas fisik, gas = EUI x luas lantai.
- Data test diambil acak per baris, sehingga informasi tingkat wilayah dapat dipakai secara sah.

Di atas fondasi itu kami membangun:
- 34 anggota model cross-validated (XGBoost, LightGBM, dan beberapa varian MLP);
- blend non-negatif per kolom yang dinilai secara nested;
- koreksi residual berbasis kemiripan stok bangunan di sekitar setiap kawasan.

Skor validasi silang nested kami 0,2496 sebelum koreksi residual (sekitar 0,2489 setelah koreksi, diestimasi di holdout internal), dan public leaderboard 0,24943. Sepanjang sepuluh submisi, public LB selalu sama dengan skor CV (ditambah estimasi holdout untuk post-processing) plus selisih tetap 0,0005 (simpangan baku 0,00004). Seluruh pipeline deterministik: run bersih dan eksekusi notebook menghasilkan file submisi dengan sha256 yang sama.

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

Metrik lomba adalah rata-rata RMSE tiap target dibagi simpangan baku bekunya. Karena dihitung per kolom, setiap target boleh punya pipeline sendiri, dan meminimalkan MSE tiap kolom sama dengan meminimalkan metrik. Prediksi konstan memberi skor sekitar 1,0 per target. Pada model final kami, EUI menyumbang sekitar 52% galat, listrik 25%, gas 20%, dan adopsi 3%. Prioritas kerja kami mengikuti porsi itu.

## 2. Data dan eksplorasi
Train berisi 429.303 baris dan test 183.987 baris. Ada 201 fitur anonim dalam lima keluarga (dimensi bangunan, era konstruksi, bauran pemanas, iklim, sosio-demografi) ditambah dua ID spasial: 330 zona dan sekitar 150 ribu subzona.

Temuan yang paling menentukan arah pemodelan:
1. **Identitas eksak.**
   - EUI dikali `bld_dim_06` sama dengan gas sampai presisi mesin (selisih relatif maksimum 7e-16). Jadi `bld_dim_06` adalah luas lantai, dan satu model gas sekaligus memberi EUI.
   - Rasio bangunan bersih sama persis dengan round(100 x `bld_dim_02`/`bld_dim_01`, 2) di 100% baris. Target ini dihitung, bukan diprediksi.
2. **Adopsi pemanas listrik konstan di dalam subzona** (subzona adalah unit sensus). Bauran pemanas `hvac_dist_*` adalah pecahan cacah per jumlah rumah tangga N, sehingga N bisa direkonstruksi eksak sebagai penyebut terkecil yang membuat semua pecahan bulat. Adopsi sama dengan sisa kategori pemanas dikurangi satu kategori yang hilang.
3. **Target makro didominasi ukuran kawasan.** Korelasi Spearman gas dengan luas lantai 0,92, dan listrik dengan jumlah meter 0,90. Sebaliknya, EUI hanya berkorelasi lemah dengan fitur tunggal (maksimum 0,18).
4. **Noise label besar dan terstruktur.** Varians residual log mengikuti pola a + b/M, dengan M jumlah meter (Gambar 2). Pola ini khas rata-rata konsumsi rumah tangga: bila konsumsi tiap rumah tangga bervariasi acak, varians relatif totalnya turun sebanding 1/M. Pada blend final, 73% varians residual gas dan 77% varians residual listrik adalah noise semacam ini, yang tidak bisa dijelaskan fitur apa pun. Lantai skor dari fitur saja sekitar 0,21.

## 3. Strategi validasi
Sebelum memilih skema cross-validation, kami memeriksa bagaimana data test dibuat:
- Rasio train/total tepat 0,700000.
- Proporsi test per zona tersebar persis binomial (std z-score 0,989).
- Porsi baris test di subzona yang tidak muncul di train adalah 8,14%, hampir sama dengan harapan split acak per baris (8,17%).
- Adversarial validation (LightGBM) memberi AUC 0,50, artinya train dan test tidak bisa dibedakan.

Kesimpulannya, test adalah sampel acak 30% per baris. Karena itu CV utama memakai KFold acak per baris (5 dan 10 fold, seed 42), dan statistik tingkat wilayah (target encoding subzona, zona, jendela spasial) sah dipakai selama dihitung out-of-fold. [[CV sekunder per zona: hasil E0271-E0278, seberapa besar skor bergantung pada wilayah yang terlihat di train.]]

Bukti bahwa CV ini bisa dipercaya datang dari leaderboard (Gambar 3). Pada sepuluh submisi, public LB sama dengan CV nested ditambah estimasi holdout untuk post-processing, plus selisih tetap 0,00052 (simpangan baku 0,00004). Urutan kandidat di CV selalu sama dengan urutan di LB.

## 4. Rekayasa target dan fitur
**Formulasi target.** Karena gas = EUI x luas, semua model keluarga gas memprediksi kuantitas per unit lalu dikalikan kembali. Pilihan unitnya beragam:
- per meter;
- per luas lantai;
- per luas selubung (GEA);
- per jumlah sertifikat energi;
- logaritmanya.

Bobot sampel diatur agar loss setara dengan MSE pada skala asli (misalnya bobot M²/A untuk gas per meter). Keragaman formulasi ini ternyata sumber keragaman ensemble terbesar. Kami juga menguji objektif Poisson (GLM laju dengan eksposur). Pada model tunggal GBM gain-nya besar (EUI 0,0039, gas 0,0025), tetapi di blend hanya 0,00004 karena ensemble sudah menyerap efek yang sama, sehingga tidak kami masukkan ke pipeline final.

**Fitur.**
- Rasio fisik (luas per meter, luas per sertifikat, faktor bentuk).
- Agregat transduktif per subzona dan zona dari baris train dan test (jumlah baris, total luas, porsi baris). Tidak memakai target.
- Target encoding out-of-fold untuk subzona, zona, jendela ±2 dan ±10 nomor subzona (yang berurutan secara spasial), serta sel grid iklim di luar subzona sendiri.
- **Fitur kNN-target (FB07):** rata-rata target dari k kawasan dengan stok bangunan paling mirip dalam jendela spasial. Pada XGBoost tunggal fitur ini menurunkan NRMSE gas 0,0011 dan EUI 0,0022, membaik di kelima fold.

Semua fitur bertarget dihitung dengan cross-fitting dan lulus uji kebocoran: label baris validasi diacak, lalu fitur validasi diperiksa tetap identik.

## 5. Pemodelan
Tabel 1 merangkum 34 anggota blend final. Satu MLP multi-task mempelajari gas dan listrik sekaligus, sehingga jumlah jaringan yang dilatih 33.

**Tabel 1.** Anggota blend final per keluarga (NRMSE OOF; bobot = porsi bobot NNLS).

| Keluarga | Anggota gas/EUI | Anggota listrik | Terbaik EUI (gas) | Terbaik listrik | Bobot gas / EUI / listrik |
|---|---|---|---|---|---|
| XGBoost (CUDA) | 12 | 5 | 0,6598 (0,2504) | 0,3221 | 43% / 52% / 20% |
| LightGBM | 4 | 3 | 0,6602 (0,2504) | 0,3221 | 11% / 11% / 17% |
| MLP | 2 | 1 | 0,6677 (0,2530) | 0,3213 | 8% / 4% / 6% |
| MLP + embedding PLR | 3 | 1 | 0,6594 (0,2489) | 0,3189 | 29% / 19% / 18% |
| TabM-mini | 0 | 1 | | 0,3199 | 0 / 0 / 11% |
| MLP multi-task | 1 | 1 | 0,6593 (0,2490) | 0,3187 | 9% / 14% / 28% |

- XGBoost memakai `hist` di CUDA dan LightGBM `deterministic=True`. 27 anggota dilatih 10-fold dan 7 anggota 5-fold, dengan split fold yang sama (KFold seed 42).
- MLP (PyTorch) dilatih tiga seed dengan jumlah epoch tetap pada jadwal one-cycle, tanpa memilih epoch dari fold validasi, sehingga OOF tetap jujur. Varian PLR memakai embedding periodik untuk fitur numerik. TabM-mini memakai 16 adapter berbagi bobot.
- CatBoost GPU diuji dan kuat, tetapi tidak deterministik antar run, sehingga dikeluarkan dari pipeline final. Biayanya di CV sekitar 0,00001.

## 6. Ensemble dan koreksi residual
**Blend.** Bobot non-negative least squares di-fit per kolom yang dinilai: gas dengan skala baris luas lantai, EUI dan listrik langsung. Anggota dipilih secara greedy maju-mundur dengan skor blend nested (bobot di-fit di fold lain, dinilai di fold yang tidak ikut fitting). Stacking level-2 dengan LightGBM, blend per kelompok ukuran kawasan, dan bobot dari holdout tidak lebih baik dari NNLS (selisih paling banyak 0,00006). Blend memperbaiki anggota tunggal terbaik sebesar 0,0083 pada EUI, 0,0036 pada gas, dan 0,0027 pada listrik.

**Koreksi residual.** Residual OOF blend ternyata masih berkorelasi di antara kawasan yang berdekatan dan mirip stok bangunannya. Prediksi test digeser oleh kombinasi linear rata-rata residual (di-shrink n/(n+k)) dari:
- zona yang sama dan sel iklim yang sama;
- jendela ±30 nomor subzona;
- k kawasan dengan stok bangunan paling mirip dalam jendela spasial (lima sampai enam konfigurasi ruang kemiripan, lebar jendela, dan k per metrik).

Bobot kombinasinya di-fit pada holdout internal 70/30: baris holdout berperan sebagai baris test, dan tetangganya hanya diambil dari bagian 70% dengan residual OOF yang tidak pernah melihat label holdout. Bobot tidak di-fit di OOF penuh, karena di sana residual tetangga sudah tercemar lewat target encoding. Gain di holdout: gas 0,24643 → 0,24539, EUI 0,65115 → 0,64961, listrik 0,31508 → 0,31477, atau sekitar 0,0006 pada skor total (Gambar 5).

**Adopsi dan duplikat.** Untuk subzona yang ada di train, nilai adopsi disalin. Untuk 8% baris di subzona baru, cacah kategori pemanas yang hilang dimodelkan sebagai binomial dengan LightGBM (objektif cross-entropy, bobot cacah, fitur N dan proporsi tetangga). Terakhir, 77 baris test identik dengan baris train pada subzona, jumlah meter, jumlah sertifikat, luas lantai, dan satu fitur sensus yang hampir unik. Semua 92 grup duplikat di train punya target identik, sehingga target baris itu disalin.

## 7. Hasil dan ablation
**Tabel 2.** Perkembangan dari baseline ke model final (NRMSE, CV primer 5-fold untuk baseline, nested 10-fold untuk final).

| Model | t1 gas | t2 listrik | t3 EUI | t4 bersih | t5 adopsi | Rata-rata |
|---|---|---|---|---|---|---|
| B0 rata-rata | 1,0000 | 1,0000 | 1,0001 | 1,0001 | 1,0007 | 1,0002 |
| B1 Ridge | 0,3114 | 0,3556 | 0,7929 | 0,2648 | 0,2759 | 0,4001 |
| B4 XGBoost | 0,2631 | 0,3301 | 0,6865 | 0,0158 | 0,0950 | 0,2781 |
| B6 MLP multi-output | 0,2654 | 0,3276 | 0,6903 | 0,0539 | 0,1478 | 0,2970 |
| Final: blend nested | 0,2452 | 0,3161 | 0,6510 | 0 | 0,0357 | 0,2496 |
| Final + koreksi residual (estimasi holdout) | | | | | | 0,2489 |

Public LB model final: 0,24943.

Ablation model tunggal yang paling berpengaruh (Gambar 5):
- 10 fold dibanding 5 fold: EUI -0,0018, gas -0,0010 (lebih banyak data latih dan encoding subzona yang lebih kaya).
- Target encoding sel grid iklim di luar subzona sendiri: EUI -0,0012.
- Agregat transduktif subzona/zona: EUI -0,0011.
- Fitur kNN-target: EUI -0,0022, gas -0,0011.
- Objektif Poisson: EUI -0,0039, gas -0,0025, listrik -0,0025 (model tunggal), tetapi blend hanya -0,00004.

Yang tidak berhasil (semua tercatat di registry):
- Stacking level-2, blend per ukuran kawasan, dan blend terpisah untuk subzona baru.
- Loss Poisson pada MLP (EUI memburuk 0,0045).
- 20 fold dibanding 10 fold: OOF terlihat membaik, tetapi holdout tidak.
- Koreksi residual lintas target (residual listrik untuk gas dan sebaliknya): gain nol.
- Pencarian kebocoran (urutan ID, nilai gas "kembar", relasi rasio antar kolom): hanya 77 baris duplikat yang bisa dimanfaatkan.

## 8. Interpretasi kebijakan
Kami menghitung TreeSHAP pada 30 ribu baris validasi untuk model LightGBM per target (Gambar 6 sampai 8): log EUI (t3, sekaligus t1 = EUI x luas), log listrik per meter (t2), dan adopsi (t5). Rasio bangunan bersih (t4) adalah identitas eksak, jadi tidak perlu model. Nama fitur dianonimkan panitia, sehingga makna kolom di bawah ini kami simpulkan dari sebaran nilainya dan kami tulis sebagai dugaan.

Porsi |SHAP| per keluarga fitur:

| Target | Rasio fisik | Sosio-demografi | Target encoding spasial | Dimensi bangunan | Era bangunan | Bauran pemanas |
|---|---|---|---|---|---|---|
| t3 EUI | 25% | 25% | 19% | 10% | 11% | 2% |
| t2 listrik per meter | 29% | 35% | 13% | 11% | 5% | 2% |
| t5 adopsi | 45% (sisa pangsa pemanas) | 21% | 3% | 1% | 0% | 27% |

**Dilema 1: skala vs intensitas termal (t1 vs t3).** Gas total terutama mengikuti ukuran kawasan (luas lantai dan jumlah meter). Sebaliknya, EUI turun tajam seiring luas per meter: kawasan dengan hunian kecil (sekitar 20 m² per meter) punya intensitas gas per m² lebih dari dua kali kawasan dengan hunian besar (200 m² per meter). Porsi bangunan era tertentu (`bld_era_09` dan `bld_era_15`) juga masuk fitur teratas. Implikasinya, program yang mengejar penurunan emisi absolut dan program yang mengejar efisiensi per m² akan memilih kawasan yang berbeda. Daftar prioritas sebaiknya dibuat terpisah untuk kedua tujuan itu.

**Dilema 2: beban jaringan listrik (t2 dan t5).** Listrik per meter naik hampir linear dengan luas hunian per meter. Fitur sisa pangsa pemanas (100 dikurangi semua kategori `hvac_dist`, yang sebagian besar adalah pemanas listrik) juga termasuk pendorong utama. Kawasan dengan hunian besar dan pemanas listrik tinggi adalah kawasan dengan beban per meter terbesar, sehingga paling rawan kemacetan gardu ketika elektrifikasi pemanas dipercepat.

**Dilema 3: kesenjangan adopsi (t5).** Adopsi pemanas listrik paling ditentukan oleh infrastruktur pemanas yang sudah ada. Setelah sisa pangsa pemanas, pendorong terkuat adalah `hvac_dist_02` (rata-rata 82%, kemungkinan pangsa pemanas gas sentral). Di kawasan dengan pangsa ini di bawah 40%, kontribusinya ke adopsi mencapai +5 sampai +15 poin persen, sedangkan di kawasan yang hampir sepenuhnya memakai gas kontribusinya negatif. Faktor sosio-demografi baru menyusul (21% porsi |SHAP|). Artinya, adopsi saat ini terkonsentrasi di kawasan yang memang belum terhubung kuat ke jaringan gas. Mendorong elektrifikasi di kawasan yang sangat bergantung pada gas butuh insentif yang berbeda dari kawasan yang sudah terbiasa dengan pemanas listrik.

Keluarga sosio-demografi menyumbang seperempat sampai sepertiga porsi |SHAP| untuk EUI dan listrik. Konsumsi energi kawasan tidak hanya soal fisik bangunan, tetapi juga komposisi penghuninya.

## 9. Reproducibility
- Seed dikunci di semua tahap (split, model, PyTorch), tidak ada batas waktu training, urutan kolom diambil dari list yang di-sort, dan versi paket dipin di `final/requirements.txt`.
- Determinisme diuji per keluarga model: XGBoost CUDA, LightGBM CPU, dan semua varian MLP memberi hasil identik byte antar run. CatBoost GPU tidak, sehingga dikeluarkan.
- `final/run_all.py` adalah satu pintu dari data mentah ke submisi. Notebook utama menyematkan seluruh kode sumber yang sama dan menjalankannya dari kondisi bersih.
- Run bersih dan eksekusi notebook memberi sha256 [[SHA]], identik dengan submisi yang diunggah.
- Notebook kedua membuat ulang submisi yang sama dari model tersimpan (dengan checksum) dalam waktu jauh lebih singkat, sebagai cadangan bila juri tidak bisa melatih ulang penuh.

## 10. Keterbatasan dan pengembangan
- Sekitar 75% varians residual gas dan listrik berasal dari rata-rata rumah tangga yang tidak teramati. Perbaikan berikutnya butuh data per rumah tangga, bukan model yang lebih besar.
- Interval prediksi per kawasan bisa diturunkan langsung dari pola a + b/M. Ini berguna bagi perencana jaringan untuk mengetahui kawasan mana yang prediksinya kurang pasti.
- Pipeline penuh butuh [[jam]] di satu GPU RTX A4000. Untuk penggunaan operasional, subset anggota dengan bobot terbesar memberi hampir seluruh skor dengan waktu jauh lebih singkat.

## 11. Kesimpulan
Skor kami datang dari tiga keputusan. Pertama, mencari struktur fisik sebelum memodelkan: dua target selesai tanpa model, dan gas serta EUI berbagi satu model. Kedua, membangun validasi yang meniru cara test dibuat, lalu membuktikannya di leaderboard: selisih CV dan LB tetap di sepuluh submisi. Ketiga, menilai setiap komponen secara jujur, dari bobot blend yang nested sampai bobot koreksi residual yang di-fit di holdout. Dengan sekitar tiga perempat galat tersisa berupa noise rata-rata rumah tangga, kami yakin skor ini dekat dengan batas yang bisa dicapai dari data yang tersedia.
