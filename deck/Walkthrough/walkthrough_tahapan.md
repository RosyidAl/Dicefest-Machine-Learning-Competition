# Walkthrough Tahapan: dari Kerangka Masalah ke Model Final

Dokumen ini mencatat tahapan proyek DICE FEST 2026 secara berurutan. Setiap tahap berisi:
- pertanyaan yang dijawab;
- bukti dari data atau eksperimen;
- keputusan yang diambil;
- alasan di balik keputusan itu.

Sumber angka: `reports/eda/EDA_REPORT.md`, `reports/DECISIONS.md`, `reports/COVERAGE.md`, dan `experiments/registry.csv`.

## Ringkasan tahapan

| Tahap | Pertanyaan | Keputusan utama |
|---|---|---|
| 1. Kerangka | Apa yang diukur dan apa yang diprioritaskan? | Pipeline terpisah per target; fokus ke target dengan NRMSE terbesar |
| 2. Target | Adakah target yang ditentukan rumus? | Gas dan EUI dari satu model, porsi bangunan bersih dengan rumus, adopsi disalin per subzona |
| 3. Kualitas data | Seberapa bersih data, dan apa arti kolom yang disamarkan? | Buang kolom konstan, salin target 77 duplikat, temukan kolom luas dan meter |
| 4. Validasi | Bagaimana data uji dibuat? | KFold acak per baris; GroupKFold zona hanya sebagai pembanding |
| 5. Baseline dan batas | Seberapa jauh galat bisa turun? | Batas bawah skor sekitar 0,21; perbaikan dicari di formulasi dan ensemble |
| 6. Formulasi dan fitur | Dalam bentuk apa target dan fitur diberikan ke model? | Target per meter/luas, fitur fisik, rata-rata wilayah dan kNN *out-of-fold* |
| 7. Model | Keluarga model mana yang dipakai? | XGBoost, LightGBM, MLP (PLR, TabM-mini, multi-task), NCA; CatBoost dibuang |
| 8. Ensemble dan final | Bagaimana model digabung dan dipilih? | Blend NNLS berbobot adaptif, koreksi residual spasial, satu berkas final yang reprodusibel |
| 9. Kebijakan | Bagaimana model menjadi rekomendasi? | Model penjelas terpisah, SHAP, empat kuadran zona |

## Tahap 1. Kerangka masalah

**Pertanyaan.** Masalah apa yang diselesaikan, dan bagaimana keberhasilan diukur?

**Bukti.**
- Satu baris adalah agregat satu kawasan, bukan satu rumah tangga: 429.303 baris latih dan 183.987 baris uji, dengan 330 zona.
- Skor = rata-rata RMSE_k / σ_k untuk lima target, dengan σ = simpangan baku data latih yang dibekukan.
- Prediksi rata-rata memberi sekitar 1,0 per target (E0169: 1,00000; 1,00000; 1,00006; 1,00008; 1,00067).
- 1% baris teratas menyumbang 14% sampai 35% dari jumlah kuadrat deviasi.

**Keputusan.**
- Setiap target diberi pipeline sendiri.
- Evaluasi selalu di skala asli.
- Tenaga dicurahkan ke target dengan NRMSE terbesar.

**Alasan.**
- Metrik dihitung per kolom, jadi tidak ada keuntungan memakai satu model *multi-output*.
- Kontribusi tiap target ke skor adalah NRMSE_k / 5. Perbaikan 0,001 pada satu target hanya mengubah skor 0,0002.
- Juri final menilai kebijakan (35 poin) dan fitur (25 poin), jadi model juga harus bisa dijelaskan.

**Asumsi yang ditolak.** "*Public LB* adalah ukuran kemajuan." Public LB hanya mencakup 30% data uji (sekitar 55 ribu baris) dan dibatasi 5 submisi per hari. Seleksi model memakai CV.

## Tahap 2. Memahami target

**Pertanyaan.** Apa arti kelima target, dan adakah relasi pasti di antaranya?

**Bukti.**
- Gas = EUI × `bld_dim_06`, dengan selisih relatif maksimum 6,7e-16. Jadi `bld_dim_06` adalah luas lantai.
- Porsi bangunan bersih = round(100 × `bld_dim_02` / `bld_dim_01`, 2), tepat di 100% baris latih.
- Adopsi pemanas listrik konstan di setiap subzona, dan selalu ≤ sisa porsi `hvac_dist` (R² linear 0,978).
- Gas terpotong di bawah 1 juta kWh (maksimum 999.765). Adopsi punya 33.859 nilai nol.
- Korelasi gas dengan listrik 0,887; gas dengan EUI hanya 0,18.

**Keputusan.**
- Gas dan EUI berasal dari satu keluarga model.
- Porsi bangunan bersih dihitung dengan rumus (NRMSE 0).
- Adopsi disalin dari subzona yang sudah dikenal. Model hanya dipakai untuk sekitar 8% baris uji di subzona baru.

**Alasan.**
- Relasi pasti menghapus kebutuhan akan model.
- Dua target selesai sebelum pemodelan, sehingga masalah ML tersisa tiga: gas/EUI, listrik, dan adopsi subzona baru.

## Tahap 3. Kualitas dan struktur data

**Pertanyaan.** Seberapa bersih data, dan apa arti kolom yang disamarkan?

**Bukti.**
- Ada dua kolom teks: `bld_dim_18` (empat kategori ordinal) dan `socio_demog_016` (Urban/Rural). Keduanya di-*encode* ke angka.
- NaN hanya ada di `bld_dim_04` dan `bld_dim_07` (695 baris latih, 336 baris uji). Ada satu nilai sentinel 9999 di `bld_dim_03`.
- Ada 11 kolom konstan dan 6 pasangan kolom dengan |r| > 0,999.
- Tidak ada baris duplikat persis. Namun 77 baris uji identik dengan baris latih pada kunci gabungan, dan di data latih grup dengan kunci yang sama selalu bertarget identik.
- 113 dari 132 kolom sensus konstan di dalam subzona, jadi kolom-kolom itu atribut subzona, bukan atribut kawasan.
- `bld_dim_12` adalah jumlah meter. Rasio listrik/`bld_dim_12` paling stabil (log-std 0,318).
- `hvac_dist` tidak berjumlah 100, dan jumlah rumah tangga N bisa direkonstruksi tepat dari kolom itu.
- `env_climate_01/03/05` membentuk sel grid iklim, sekitar 15 baris latih per sel.

**Keputusan.**
- Buang kolom konstan.
- Biarkan NaN untuk model pohon.
- Salin target untuk 77 baris duplikat (`dup_fix`).
- Pakai luas lantai dan jumlah meter sebagai pembagi (*exposure*).
- Pakai N untuk model adopsi.

**Alasan.**
- Model berbasis pohon menangani NaN dan kolinearitas tanpa imputasi.
- Menemukan arti kolom yang disamarkan membuka formulasi target per unit di Tahap 6.

## Tahap 4. Desain validasi

**Pertanyaan.** Bagaimana data uji dibuat, dan skema validasi apa yang menirunya?

**Bukti.**
- Porsi data uji tepat 0,300000 dari total baris.
- Semua zona di data uji ada di data latih. 91,86% baris uji berada di subzona yang dikenal, sangat dekat dengan harapan split acak per baris (8,14% teramati vs 8,17% harapan untuk subzona tak dikenal).
- Rasio uji per zona menyebar seperti binomial (std z-score 0,989).
- *Adversarial validation* memberi AUC 0,4997, artinya data latih dan data uji tidak bisa dibedakan.

**Keputusan.**
- CV utama: KFold acak per baris (5 fold untuk eksperimen, 10 fold untuk anggota final).
- CV sekunder: GroupKFold per zona, hanya untuk dilaporkan.

**Alasan.**
- Validasi harus meniru cara data uji dibuat.
- Pada GroupKFold zona, rata-rata target wilayah tampak merusak (EUI 0,833 dengan fitur itu vs 0,692 tanpa). Pada KFold acak, fitur yang sama menurunkan EUI 0,0022.
- Selisih CV dan *public LB* stabil di sekitar +0,0005 pada banyak submisi, dan urutan antarsubmisi selalu sama.

**Asumsi yang ditolak.** "Data spasial harus divalidasi per wilayah." Anjuran itu hanya berlaku bila data uji dibuat per wilayah, sedangkan di sini data uji dibuat acak per baris.

## Tahap 5. Baseline dan batas akurasi

**Pertanyaan.** Seberapa jauh galat bisa diturunkan?

**Bukti.**

| Baseline | Skor CV |
|---|---|
| B0 rata-rata | 1,00016 |
| B1 Ridge | 0,40011 |
| B6 MLP sederhana | 0,29699 |
| B4 XGBoost default | 0,27808 |

- Varians residual log mengikuti a + b/M, dengan M = jumlah meter. Untuk gas, a = 0,0091 dan b = 0,377; untuk listrik, a = 0,0113 dan b = 0,591.
- Sekitar 75% varians residual berasal dari rata-rata antarrumah tangga yang tidak tercatat.
- Bahkan bila bagian galat model (a) bernilai 0, skor total sekitar 0,213.

**Keputusan.**
- Target realistis: perbaikan kecil dan bertahap di atas sekitar 0,25.
- Perbaikan dicari di formulasi target, fitur spasial, dan ensemble, bukan di model yang lebih besar.

**Alasan.**
- Batas bawah ini menjelaskan mengapa skor di bawah 0,2 tidak bisa dicapai dari fitur yang tersedia.
- Galat model (a) datar di semua ukuran kawasan, jadi tidak ada segmen yang tertinggal untuk dieksploitasi.

## Tahap 6. Formulasi target dan fitur

**Pertanyaan.** Dalam bentuk apa target dan fitur sebaiknya diberikan ke model?

**Formulasi target.**
- Gas dan listrik diprediksi per unit, lalu dikalikan kembali dengan unitnya: per meter, per luas, per jumlah sertifikat (EPC), atau per `bld_dim_03`.
- Bobot berupa pangkat *exposure*, supaya *loss* setara dengan MSE di skala asli.
- Ada varian log (dengan koreksi bias) dan varian Poisson.
- Alasan: total kawasan terutama ditentukan ukurannya. Yang perlu dipelajari model adalah perilaku per unit.

**Fitur.**
- **Rasio fisik:**
  - `area_per_meter` (luas rata-rata per rumah, fitur SHAP terbesar);
  - `area_per_epc`;
  - `gea_ratio`;
  - `hvac_rest`.
- **Agregat subzona dan zona** dari data latih + uji tanpa target: LightGBM gpm EUI -0,0011.
- **Rata-rata target *out-of-fold*** per subzona, zona, jendela `sub_n`, dan sel iklim. Sel iklim saja: XGBoost EUI 0,6677 → 0,6665.
- **kNN-target *out-of-fold*** (FB07): rata-rata target kawasan dengan stok bangunan termirip di dalam jendela subzona. Hasilnya gas -0,0011 dan EUI -0,0022, membaik di 5 dari 5 fold. FB07b menambah EUI -0,00045.

**Pengaman kebocoran.**
- Semua fitur yang memakai target dihitung *out-of-fold*, dengan fold dalam di setiap fold luar.
- Uji `tests/test_no_leakage.py`: label validasi diacak, dan fitur validasi/uji harus identik.

**Yang ditolak.**
- Transformasi Tweedie 1,2.
- *Loss* Poisson untuk MLP.
- Rata-rata target berbobot (*ratio-of-sums*).
- Penambahan fitur konteks yang lebih banyak.

Semua yang ditolak tidak memberi gain pada fold yang sama.

## Tahap 7. Pemilihan keluarga model

**Pertanyaan.** Keluarga model mana yang cocok untuk tiap target?

**Bukti.**
- Model tunggal terbaik:
  - EUI: xp_gpm (XGBoost Poisson, gas per meter), 0,65741;
  - listrik: p3_epm_wM (MLP PLR), 0,31835.
- CatBoost GPU tidak deterministik. *Rerun* memberi iterasi berbeda dan selisih prediksi 352 kWh. Membuangnya hanya menambah CV +0,00001.
- MLP PLR p1_lepm lebih lemah sebagai model tunggal (0,32174 vs 0,32035) tetapi memperbaiki blend listrik -0,00056.
- MLP multi-task m1 adalah anggota EUI tunggal terbaik saat itu (0,66077), tetapi menambah blend hanya -0,00004.

**Keputusan.**
- Pipeline final hanya memakai:
  - XGBoost (GPU);
  - LightGBM (CPU);
  - MLP PyTorch (polos, PLR, TabM-mini, multi-task);
  - NCA.
- Semua keluarga itu terbukti identik byte saat dijalankan ulang.

**Alasan.**
- Aturan kompetisi menuntut hasil yang identik saat diaudit.
- Anggota dipilih karena pola galatnya berbeda dari anggota lain, bukan hanya karena skor tunggalnya baik.

**Yang ditolak (tanpa gain di blend).**
- LightGBM `linear_tree` dan `extra_trees`.
- CatBoost depth 10.
- Stacking level-2 LightGBM.
- Blend per kelompok ukuran.

## Tahap 8. Ensemble, koreksi, dan pemilihan final

**Pertanyaan.** Bagaimana model digabung, dikoreksi, dan dipilih menjadi submisi final?

**Langkah dan buktinya.**

| Langkah | Bukti | Alasan |
|---|---|---|
| Blend NNLS dengan penilaian *nested* | Pemangkasan membuang 17 dari 50 anggota tanpa rugi (D005) | Bobot tidak boleh dinilai di data yang dipakai untuk mencarinya |
| Bobot adaptif A2xM2xp2 | Skor total -0,000124; EUI 0,65029 → 0,64984; *public LB* -0,00009 (D012) | Model yang paling akurat di kawasan kecil berbeda dengan model yang paling akurat di kawasan besar |
| Koreksi residual grup dan kNN | Holdout: grup EUI -0,00100, kNN tambahan EUI -0,0013 | Galat kawasan yang berdekatan dan mirip masih berkorelasi |
| λ koreksi di-fit di holdout 30% | Rata-rata *leave-one-out* pada OOF penuh bias, karena model fold lain sudah melihat target lewat fitur rata-rata wilayah | Menghindari estimasi yang terlalu optimistis |
| Anggota NCA (D014) | Total -0,000032; *public LB* -0,00003 | Membawa pola galat baru |
| S1, S3, S4 (D015) | Kawasan bermeter ≤ 3 terprediksi terlalu rendah sekitar 20%; *public LB* -0,00007 | Bias sistematis di dua batas data |
| Adopsi v2 dipertahankan (D011) | v3 lebih baik di CV tetapi lebih buruk di *public LB* | Proksi CV untuk subzona baru tidak cukup mirip |

**Pemilihan final (D-FINAL).**
- Kedua slot final diisi berkas yang sama: `sub_final_pipeline.csv` (sha256 aec745ab), *public LB* 0,24913.
- Hasil *private LB*: 0,24844, peringkat 8.
- Gabungan akhir terdiri dari 41 model: 28 untuk gas/EUI dan 13 untuk listrik.

**Alasan.**
- Hanya berkas itu yang direproduksi persis oleh notebook dan oleh run final dari nol.
- Berkas lain dengan skor publik setara (misalnya `sub_v19nca3pp`) tidak bisa direproduksi notebook. Bila berkas itu yang menang di LB privat, hasilnya akan gagal audit.

**Pola keputusan.** Setiap perubahan diterima bila:
- membaik di *nested* CV atau holdout;
- membaik di minimal 4 dari 5 fold; dan
- arah perubahannya di *public LB* sesuai perkiraan.

Perubahan yang di bawah sekitar 0,1% relatif dianggap noise, kecuali ada arahan untuk mengambil hasil maksimum (D009).

## Tahap 9. Dari model ke kebijakan

**Pertanyaan.** Bagaimana model diterjemahkan menjadi rekomendasi?

**Keputusan.**
- SHAP dihitung pada model penjelas terpisah, yaitu satu LightGBM per target yang hanya memakai fitur kawasan terukur, bukan pada gabungan 41 model.
- R² model penjelas: gas per meter 0,75, EUI 0,56, listrik per meter 0,49, adopsi 0,99.
- Zona dipetakan dengan dua ukuran, yaitu intensitas gas zona (EUI, kWh per m²) dan kenaikan listrik tahunan bila elektrifikasi penuh (%), lalu dibagi empat kuadran berdasarkan median (`explain.zone_indices`). Potensi *retrofit* (TWh) dihitung terpisah dan tidak menjadi sumbu peta.
- Uji sensitivitas memakai porsi pemanasan 70/80/90% dan COP 2,5/3/3,5.

**Alasan.**
- Gabungan 41 model dengan fitur *out-of-fold* sulit dijelaskan per variabel.
- Model penjelas memberi bacaan yang bisa ditelusuri ke karakteristik kawasan.
- SHAP menunjukkan asosiasi, bukan sebab-akibat. Karena itu rekomendasi disertai keterbatasan:
  - skenario listrik memakai energi tahunan, bukan beban puncak;
  - ambang kuadran berupa median.

## Tahap 9a. Membaca kuadran (Q) dan rekomendasi (R)

**Pertanyaan.** Apa arti Q1 sampai Q4 dan R1 sampai R3, dan bagaimana keduanya terhubung?

### Kuadran (Q): kelompok zona

Setiap zona dinilai dengan dua pertanyaan:
1. Seberapa boros gasnya? Ukurannya intensitas gas zona (EUI): total gas zona dibagi total luas lantai zona.
2. Bila semua pemanas gas diganti pompa panas, berapa persen konsumsi listrik tahunan zona naik? Rumusnya: tambahan listrik = gas × 0,8 ÷ 3, lalu dibagi listrik zona saat ini.

Setiap ukuran dibelah di median 330 zona: EUI 133,5 kWh/m² dan kenaikan listrik 94,85%. Angka 94,85% ini sama dengan angka besar di *slide* 1.

| | Kenaikan listrik ≥ median | Kenaikan listrik < median |
|---|---|---|
| **EUI ≥ median** | **Q1**: 118 zona, 46,8% gas | **Q2**: 47 zona, 11,5% gas |
| **EUI < median** | **Q3**: 47 zona, 14,6% gas | **Q4**: 118 zona, 27,1% gas |

Median per kuadran (dihitung dari `reports/policy/zones.csv`):

| Kuadran | EUI (kWh/m²) | Kenaikan listrik |
|---|---|---|
| Q1 | 144,4 | 102,0% |
| Q2 | 139,1 | 90,1% |
| Q3 | 128,4 | 97,3% |
| Q4 | 119,9 | 88,5% |

Catatan penting:
- **"Di bawah median" tidak sama dengan "kecil".** Kenaikan listrik di Q2 masih sekitar +90%. Di depan juri, ucapkan "di bawah median" atau "relatif lebih rendah", jangan "kecil".
- **Kenaikan dihitung dalam persen dari listrik saat ini.** Zona yang sudah memakai banyak listrik mendapat persentase kenaikan yang lebih rendah untuk tambahan yang sama.
- **Q1 dan Q4 masing-masing 118 zona, Q2 dan Q3 masing-masing 47.** Artinya zona yang boros gas cenderung juga mengalami kenaikan listrik tinggi.

### Rekomendasi (R): program untuk kuadran tertentu

| | Kuadran | Isi | Alasan | Zona contoh |
|---|---|---|---|---|
| R1 | Q1 | *Retrofit* selubung (dinding, atap, jendela) dulu, pompa panas kemudian | Gas boros dan kenaikan listrik tinggi; kebutuhan panas dikurangi dulu supaya tambahan listrik nanti lebih kecil | REG_273, REG_270, REG_045 (audit energi) |
| R2 | Q1 dan Q3 | Kapasitas gardu dinilai sebelum subsidi pompa panas dibuka | Kenaikan listrik di atas median | REG_263, REG_045, REG_270 |
| R3 | Q2 | Pilot pompa panas | Gas boros (penurunan emisi besar), kenaikan listrik di bawah median | REG_033 (gas jaringan 83,21%, adopsi 4,99%) |

Q4 tidak mendapat program khusus karena kedua ukurannya di bawah median, sehingga bukan prioritas awal. Kalimat ini tidak ada di *deck*; siapkan sebagai jawaban lisan.

## Tahap 9b. Hal terkait yang perlu dipahami sebelum presentasi

1. **Dari mana nilai zona berasal.** Nilai zona adalah jumlah dari semua kawasannya. Untuk kawasan latih dipakai nilai aktual, untuk kawasan uji dipakai prediksi model. Sekitar 30% nilai zona berasal dari prediksi. Peta ini juga bukan peta geografis karena data tidak memuat koordinat; letak titik ditentukan oleh dua ukuran di atas.

2. **Asumsi 80% dan COP 3.** Diasumsikan 80% gas dipakai untuk pemanasan, dan pompa panas menghasilkan 3 kWh panas dari 1 kWh listrik (COP 3). Contohnya, zona dengan gas 300 GWh mendapat tambahan listrik 300 × 0,8 ÷ 3 = 80 GWh. Kedua angka ini asumsi, bukan hasil dari data.

3. **Energi tahunan vs beban puncak.** Semua angka listrik adalah total setahun. Gardu yang kewalahan ditentukan oleh beban puncak, yaitu pemakaian tertinggi pada jam tertentu, misalnya pagi hari di musim dingin. Data tidak memuat beban puncak maupun kapasitas gardu. Karena itu R2 hanya menyusun daftar zona yang perlu dinilai, sedangkan penilaian kapasitasnya tetap tugas operator jaringan.

4. **Persen vs absolut di R2.** Ada dua cara mengurutkan zona: kenaikan dalam persen atau tambahan dalam GWh. Saat COP dibuat berbeda per iklim (2,5 sampai 3,5), urutan persen berubah besar; REG_290 turun dari peringkat 1 ke 47. Urutan absolut stabil (REG_263 tetap 1, REG_045 tetap 3, REG_270 tetap 4). Karena itu titik awal R2 dipilih dari tambahan absolut.

5. **Uji ketahanan dan Spearman.** Spearman mengukur seberapa mirip dua urutan (1 = identik).
   - Saat prediksi diberi derau acak, urutan zona hampir tidak berubah (Spearman 0,998, median 8 zona pindah kuadran).
   - Saat COP dibuat berbeda per iklim, urutannya cukup berubah (Spearman 0,81, 68 zona pindah kuadran).
   Kesimpulannya, peta ini tahan terhadap galat model tetapi peka terhadap asumsi COP. Karena itu daftar zona disebut kandidat.

6. **Model penjelas.** SHAP tidak dihitung pada gabungan 41 model, tetapi pada satu LightGBM per target yang hanya memakai karakteristik kawasan terukur. Fitur seperti rata-rata target wilayah membantu skor tetapi tidak bisa diubah lewat kebijakan, jadi tidak dipakai di model penjelas. Konsekuensinya, akurasi model penjelas lebih rendah (R² gas per meter 0,75, EUI 0,56, listrik per meter 0,49).

7. **Arti "+65%" dan "+75%".** Keduanya selisih sumbangan SHAP antara kawasan rumah terkecil dan terbesar pada model penjelas.
   - R1: sumbangan pada gas per m² di rumah terkecil sekitar 65% lebih tinggi daripada di rumah terbesar (antarkuartil +34%). Dugaannya, rumah kecil punya dinding dan atap yang relatif luas dibanding lantainya.
   - R2: sumbangan pada listrik per rumah di rumah terbesar sekitar 75% lebih tinggi daripada di rumah terkecil.
   Keduanya asosiasi, bukan sebab-akibat.

8. **Indikator *retrofit* 7,39 TWh.** Ini adalah jumlah gas yang berkurang bila sumbangan SHAP stok bangunan yang di atas rata-rata diturunkan ke rata-rata. Sebagai pembanding, total gas 141,35 TWh, dan bagian Q1 3,65 TWh. Angka ini indikator, bukan janji penghematan, karena acuannya rata-rata nasional dan biasnya bisa ke dua arah.

9. **Adopsi dan gas jaringan (R3).** Porsi jenis pemanas di satu kawasan selalu berjumlah 100%. Kalau porsi gas jaringan tinggi, sisa porsi untuk pemanas listrik otomatis kecil.
   - Adopsi pemanas listrik 9,46% di kawasan dengan gas jaringan 60–80%.
   - Adopsi hanya 2,73% di kawasan dengan gas jaringan di atas 80%, padahal kelompok ini 70% kawasan.
   Karena itu pola ini disebut pola komposisi, bukan sebab-akibat.

### Cek diri (jawab tanpa melihat)
1. Mengapa R3 dimulai di Q2, bukan di Q1?
   Jawaban: gas Q2 boros, jadi penurunan emisinya besar, sedangkan kenaikan listriknya di bawah median, jadi risiko ke gardu lebih rendah. Q1 kenaikan listriknya di atas median, sehingga kebutuhan panasnya dikurangi dulu lewat *retrofit*.
2. Mengapa R2 mencakup Q3 padahal gas Q3 tidak boros?
   Jawaban: R2 hanya melihat kenaikan listrik, dan Q3 berada di atas median pada ukuran itu.
3. Mengapa titik awal R2 dipilih dari GWh, bukan dari persen?
   Jawaban: urutan GWh tetap stabil saat COP diubah per iklim, sedangkan urutan persen tidak.
4. Apakah SHAP membuktikan rumah kecil menyebabkan boros gas?
   Jawaban: tidak. SHAP menunjukkan asosiasi pada prediksi model, jadi langkah R1 dimulai dengan audit energi dan pilot dengan pembanding.
5. Apakah daftar zona bisa langsung dipakai untuk membangun gardu?
   Jawaban: tidak. Angkanya energi tahunan dan peka terhadap COP, jadi daftar ini adalah kandidat untuk dinilai operator.
