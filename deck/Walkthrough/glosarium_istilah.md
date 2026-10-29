# Glosarium Istilah Deck v5

Daftar istilah asing dan teknis yang muncul di *deck* v5 (`deck/ppt04/BISMILLAH_Final_v5.pptx`), di layar maupun di catatan pembicara. Setiap istilah ditulis dengan arti sederhana dan tempat istilah itu muncul. "S" berarti *slide* utama, "C" berarti *slide* cadangan (C1 = *slide* 14, ..., C12 = *slide* 25).

Nama kolom data disamarkan oleh panitia, jadi makna kolom seperti `bld_dim_06` adalah dugaan tim berdasarkan hubungan antarkolom.

## 1. Energi dan kebijakan

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| Trilema energi | Tiga tujuan yang saling tarik: emisi turun, biaya terjangkau, dan jaringan listrik andal. Mengejar satu tujuan bisa merugikan yang lain | S1, judul tema lomba |
| Elektrifikasi | Mengganti peralatan berbahan bakar (di sini pemanas gas) dengan peralatan listrik | S1, S11 |
| Pompa panas (*heat pump*) | Pemanas listrik yang memindahkan panas dari udara luar ke dalam rumah, seperti kulkas yang dibalik | S1, S2, S10, S11 |
| COP (*Coefficient of Performance*) | Panas yang dihasilkan per 1 kWh listrik. COP 3 berarti 1 kWh listrik menghasilkan 3 kWh panas. Asumsi kami 3, diuji juga 2,5 sampai 3,5 per iklim | S1, S8, S10, S12, C10 |
| Asumsi 80% | Porsi gas yang diasumsikan dipakai untuk pemanasan. Sisanya dianggap untuk keperluan lain dan tidak diganti pompa panas | S1, S8 |
| Gas jaringan | Gas yang dialirkan lewat pipa ke rumah. Porsinya adalah persentase rumah tangga yang memanaskan rumah dengan gas pipa | S11 |
| Ketel (*boiler*) | Alat pemanas berbahan bakar gas. "Bantuan bongkar ketel lama" berarti membantu biaya mengganti ketel dengan pompa panas | S11 |
| kWh, GWh, TWh | Satuan energi. 1 GWh = 1 juta kWh; 1 TWh = 1.000 GWh | S3, S8, S9, S10 |
| EUI (*Energy Use Intensity*) | Konsumsi gas per m² luas lantai. Ukuran "boros" yang adil antara kawasan besar dan kecil. Juga salah satu dari 5 target | S2, S3, S6, S8 |
| Energi tahunan | Total konsumsi selama setahun (kWh). Semua angka listrik di *deck* memakai ini | S1, S8, S10 |
| Beban puncak (kW) | Pemakaian tertinggi pada satu waktu, misalnya pagi musim dingin. Inilah yang menentukan gardu kewalahan atau tidak. Tidak ada di data | S10, S12, C6 |
| Gardu (*substation*) | Fasilitas yang menurunkan tegangan listrik dan membaginya ke kawasan. Kapasitasnya terbatas | S2, S10, S12, S13 |
| Operator jaringan distribusi | Perusahaan yang mengelola kabel dan gardu ke rumah-rumah. Pelaksana R2 | S10, S12 |
| Selubung bangunan (*envelope*) | Bagian bangunan yang memisahkan dalam dan luar: dinding, atap, jendela, lantai | S9 |
| *Retrofit* selubung | Memperbaiki selubung (insulasi dinding dan atap, jendela ganda) supaya panas tidak bocor dan kebutuhan pemanas turun. Inti R1 | S2, S9, S13 |
| Indikator *retrofit* (7,39 TWh) | Perkiraan gas yang berkurang bila sumbangan SHAP stok bangunan yang di atas rata-rata diturunkan ke rata-rata. Indikator, bukan janji penghematan | S8, S9 |
| Audit energi | Pemeriksaan langsung di lapangan: di mana energi terbuang dan berapa biaya perbaikannya. Langkah pertama R1 | S9, S12, S13 |
| Pilot | Program percobaan skala kecil sebelum diperluas. Inti R3 | S2, S11, S13 |
| Zona pembanding | Zona yang tidak mendapat program, dipakai sebagai pembanding untuk melihat apakah program benar berdampak | S9, S11, S12, S13 |
| Skrining | Penyaringan awal untuk menentukan mana yang perlu diperiksa lebih lanjut. R2 adalah daftar skrining, bukan bukti kapasitas | S10 |
| Subsidi | Bantuan biaya dari pemerintah, di sini untuk membeli pompa panas | S2, S10 |
| Intensitas karbon listrik | Emisi CO₂ per kWh listrik, bergantung pada pembangkitnya. Tidak ada di data | C6 |
| Tarif | Harga listrik dan gas per kWh. Tidak ada di data | C6 |

## 2. Data dan lomba

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| Kawasan | Satu baris data, yaitu satu area kecil. Data latih berisi 429.303 kawasan, data uji 183.987 | S3–S6, S8 |
| Subzona | Kelompok beberapa kawasan. Nomor subzona yang berurutan cenderung berdekatan | S4, S6, C9 |
| Zona (`REG_xxx`) | Kelompok subzona. Ada 330 zona, dan peta kebijakan dibuat per zona | S2, S8–S11 |
| Petak iklim | Kelompok kawasan dengan kombinasi nilai kolom iklim yang sama | C2 |
| Meteran | Meteran energi. Satu meteran kira-kira satu rumah tangga | S3, S6 |
| EPC (*Energy Performance Certificate*) | Sertifikat kinerja energi bangunan. `bld_dim_01` diduga jumlah sertifikat | S3 (`area_per_epc`, `meters_per_epc`) |
| Porsi bangunan bersih | Target ke-4: persen bangunan bersertifikat yang tergolong "bersih". Tepat sama dengan 100 × `bld_dim_02` ÷ `bld_dim_01` | S6 |
| Adopsi pemanas listrik | Target ke-5: persen rumah tangga yang memakai pemanas listrik | S6, S11, C9 |
| `bld_dim_xx`, `socio_demog_xx`, `env_climate_xx`, `hvac_dist_xx`, `bld_era_xx` | Kelompok kolom yang disamarkan: dimensi bangunan, sosial-demografi, iklim, sebaran jenis pemanas, dan periode pembangunan | S3, S9 |
| Fitur | Kolom input yang dipakai model untuk menebak | S3–S5 |
| Target | Nilai yang ditebak. Ada 5: gas, listrik, EUI, porsi bangunan bersih, adopsi | S6 |
| Label | Nilai target yang sebenarnya di data latih | S5, C3 |
| Data latih dan data uji | Data latih punya label dan dipakai untuk belajar. Data uji tidak punya label dan harus ditebak | S3, S5 |
| Duplikat | 77 kawasan uji yang identik dengan kawasan latih, sehingga targetnya disalin | S7, C8 |
| Nilai hilang (NaN) | Sel data yang kosong. XGBoost dan LightGBM menerimanya apa adanya; MLP dan NCA mengisinya dengan 0 | C12 |
| Pencilan (*outlier*) | Nilai yang jauh dari kebanyakan. Tidak ada baris yang dibuang | C12 |
| Log (logaritma) | Transformasi yang meringkas rentang angka yang sangat lebar. 10 dari 41 anggota menebak log target, lalu hasilnya dikembalikan dengan exp | S3, C12 |
| Kaggle, *leaderboard* | Platform lomba dan papan skor. Publik: skor dari sebagian data uji yang terlihat selama lomba. Privat: skor dari sisa data uji yang dipakai untuk peringkat akhir | S7, C7 |
| RMSE | Rata-rata besar galat tebakan (akar dari rata-rata kuadrat galat) | S6, C1 |
| σ (sigma) | Simpangan baku target, yaitu ukuran sebaran nilainya | S6, C1 |
| RMSE/σ (NRMSE) | RMSE dibagi σ supaya kelima target bisa dibandingkan. Menebak rata-rata memberi sekitar 1, dan makin kecil makin baik. Skor lomba adalah rata-rata kelima target | S6, S7 |
| *Notebook*, sel, In [n] | Berkas Jupyter yang dikumpulkan. "In [n]" adalah nomor eksekusi sel kode yang dilihat juri | footer semua *slide* |
| Modul | Berkas `.py` yang ditulis dari dalam *notebook* (17 modul) | S6 |
| `DICEFEST_TRAIN` | Sakelar di *notebook*: latih ulang semua model, atau pakai prediksi tersimpan | S6 |
| *Writeup* | Laporan PDF yang dikumpulkan bersama *notebook* | C5, C7 |

## 3. Fitur dan pencegahan kebocoran

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| Rekayasa fitur (*feature engineering*) | Membuat kolom baru yang lebih bermakna dari kolom asli | S3–S5 |
| `area_per_meter` | Luas lantai ÷ jumlah meteran ≈ luas satu rumah. Fitur teratas | S3 |
| Rasio fisik turunan | Kelompok fitur buatan tim berupa pembagian dua kolom bangunan, misalnya `area_per_meter` dan `s002_ratio` | S3, C3 |
| Agregat kawasan | Jumlah kawasan, luas, meteran, dan sertifikat per subzona dan zona, dihitung tanpa target | C3 |
| Rata-rata target wilayah (*target encoding*) | Rata-rata target di subzona, zona, atau petak iklim yang sama, dipakai sebagai fitur | C3 |
| Kawasan termirip (kNN, *k-nearest neighbors*) | Rata-rata konsumsi dari k kawasan yang paling mirip stok bangunannya | S4, S7 |
| Jendela nomor subzona (±30, ±100, ±300) | Batas pencarian tetangga: hanya kawasan yang nomor subzonanya dekat, sebagai pengganti lokasi | S4 |
| Kebocoran (*leakage*) | Model "menyontek" jawaban, misalnya fitur yang memakai label baris itu sendiri. Skornya tampak bagus tetapi gagal di data baru | S3–S5, C3 |
| *Fold* | Satu dari beberapa bagian data. Data latih dibagi 10 (atau 5) *fold* | S4, C1 |
| Validasi silang (*cross-validation*) | Bergiliran: latih di 9 *fold*, uji di 1 *fold*, diulang untuk setiap *fold* | S4, C7 |
| *Out-of-fold* (OOF) | Nilai untuk satu *fold* dihitung hanya dari *fold* lain, jadi baris tidak pernah melihat labelnya sendiri | S4, C2 |
| Dua lapis (lapis luar, lapis dalam) | Lapis luar 10 *fold* untuk menilai model. Di dalam bagian latih dibagi lagi 5 bagian untuk menghitung fitur bertarget | S4 |
| Uji label acak | Label diacak lalu dicek apakah fitur ikut berubah. Kalau tidak berubah, berarti fitur tidak membaca label | S5, C3 |
| Kontrol | Versi yang sengaja dibuat bocor, untuk membuktikan bahwa tesnya bisa menangkap kebocoran | S5 |
| *Holdout set* | Sebagian data yang disisihkan dan tidak dipakai untuk menyusun model, hanya untuk menguji keputusan | C2, C4, C8 |
| *Overfit* | Model terlalu menghafal data latih sehingga buruk di data baru | C1, C7 |
| Validasi adversarial, AUC | Model dilatih untuk membedakan baris latih dari baris uji. AUC 0,5 berarti tidak bisa dibedakan; kami dapat 0,4997, jadi validasi kami meniru kondisi data uji | S7, C7 |
| Validasi per zona (*GroupKFold*) | Pembagian *fold* per zona, bukan per baris. Tidak dipakai karena data uji tersebar acak per baris | C7 |
| Skor jujur | Skor yang dihitung dengan bobot gabungan di-*fit* ulang di setiap *fold* luar, supaya tidak menilai di data yang dipakai untuk memilih | S6, C1 |

## 4. Model

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| Model | Program yang belajar pola dari data latih lalu menebak data uji | S6 |
| XGBoost, LightGBM | Model berbasis ratusan pohon keputusan kecil yang dibangun bertahap; tiap pohon memperbaiki kesalahan pohon sebelumnya (*gradient boosting*) | S6, C4, C5 |
| MLP (*Multi-Layer Perceptron*) | Jaringan saraf tiruan sederhana | S7, C4, C12 |
| MLP dua keluaran (mk2) | Satu MLP yang menebak gas dan listrik sekaligus | C4 |
| NCA (*Neighbourhood Components Analysis*) | Model yang belajar ukuran "kemiripan" antarkawasan, lalu menebak dari tetangga terdekat | S7, C4 |
| *Early stopping* | Pelatihan dihentikan saat skor validasi berhenti membaik, supaya tidak *overfit* | C3 |
| *Seed* | Angka awal pengacak. Dikunci supaya hasil sama setiap dijalankan | C4, C8 |
| Jalur 1, 2, 3 | Jalur 1 gas dan EUI (28 model), jalur 2 listrik (13 model), jalur 3 porsi bersih (rumus) dan adopsi (salin subzona + 2 LightGBM) | S6, C4 |
| `q_xent`, `r_reg` | Dua LightGBM adopsi untuk subzona baru. `q_xent` menebak peluang, `r_reg` menebak porsi kategori pemanas lain; adopsi adalah sisanya | C4, C9 |
| Model penjelas | Satu LightGBM per target yang hanya memakai karakteristik kawasan terukur, khusus untuk dibaca dengan SHAP. Tidak ikut submisi | S3, S9, C5 |

## 5. Penggabungan dan penyesuaian

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| *Ensemble*, *blend*, gabungan | Menggabungkan tebakan banyak model menjadi satu, dengan bobot berbeda per model | S6, S7, C1 |
| NNLS (*Non-Negative Least Squares*) | Cara mencari bobot gabungan dengan syarat semua bobot tidak negatif | C1, C12 |
| Bobot adaptif (A2xM2xp2) | Bobot tiap model berubah halus menurut peringkat luas (A), jumlah meter (M), dan prediksi global (p), masing-masing 2 simpul | S7, C1 |
| Bobot pojok, simpul | Simpul adalah titik acuan bobot. 2 × 2 × 2 = 8 bobot pojok per model, dan bobot di antaranya diinterpolasi | C1 |
| Galat (*residual*) | Selisih antara tebakan dan nilai sebenarnya | C2 |
| Koreksi galat tetangga | Bila tetangga yang mirip sama-sama ditebak terlalu rendah, tebakan kawasan ini dinaikkan sedikit | S6, C2 |
| λ (lambda) | Besar koreksi yang dipakai. Dipilih di *holdout set*, lalu dibekukan | C2 |
| n/(n + 30) | Faktor pengecil: makin sedikit tetangga (n), makin kecil koreksinya | C2 |
| Penyesuaian akhir S1, S3, S4 | S1 menaikkan tebakan kawasan 2–4 meteran, S3 menarik turun tebakan gas di dekat batas 999.765 kWh, S4 memotong ke rentang data latih | S6, S7, C8 |
| Deterministik | Dijalankan ulang di perangkat yang sama memberi hasil identik | C8 |
| sha256 | Sidik jari berkas. Kalau isinya berubah satu bit pun, sidiknya berubah. Dipakai untuk membuktikan berkas submisi identik | S7, C8 |

## 6. Interpretasi dan statistik

| Istilah | Arti sederhana | Konteks di *deck* |
|---|---|---|
| SHAP (*SHapley Additive exPlanations*) | Menguraikan tebakan model menjadi sumbangan tiap fitur: seberapa besar fitur itu mendorong tebakan naik atau turun | S3, S9–S11, C5 |
| TreeSHAP | Cara menghitung SHAP secara tepat untuk model berbasis pohon | C5 (catatan) |
| Rata-rata \|SHAP\| | Rata-rata besar sumbangan fitur, tanpa melihat arah naik atau turun. Dipakai untuk mengurutkan fitur terpenting | S3 |
| Dependensi SHAP | Grafik nilai fitur (sumbu X) terhadap sumbangan SHAP-nya (sumbu Y) | S9, S10, S11 |
| Porsi SHAP per kelompok | Persentase sumbangan total yang berasal dari satu kelompok fitur, misalnya rasio fisik 25–30% | S3, C5 |
| R² | Seberapa banyak variasi target yang dijelaskan model: 1 = sempurna, 0 = sama dengan menebak rata-rata | C5 |
| Median | Nilai tengah setelah data diurutkan. Dipakai sebagai garis pembagi kuadran | S1, S2, S8, S10, S11 |
| Kuadran Q1–Q4 | Empat kelompok zona hasil membelah peta di median intensitas gas dan median kenaikan listrik (lihat `walkthrough_tahapan.md` tahap 9a) | S2, S8–S13 |
| Rekomendasi R1–R3 | R1 *retrofit* di Q1, R2 penilaian gardu di Q1 dan Q3, R3 pilot pompa panas di Q2 | S2, S9–S11, S13 |
| Kuartil, antarkuartil | Kuartil membagi data menjadi 4 bagian sama banyak. "Antarkuartil" membandingkan 25% kawasan terkecil dengan 25% terbesar, sehingga tidak bergantung pada nilai ekstrem | S9, C11 |
| Pita 5% | 5% kawasan terkecil dibandingkan dengan 5% terbesar | S3, C11 |
| Spearman | Ukuran kemiripan dua urutan: 1 = urutan identik. 0,998 saat diberi derau, 0,81 saat COP per iklim | S8, C10 |
| Derau (*noise*) | Gangguan acak. Di uji ketahanan, prediksi diberi derau sebesar galat model untuk melihat apakah urutan zona berubah | S8, C10 |
| Uji ketahanan | Mengulang perhitungan dengan asumsi atau data yang digeser untuk melihat apakah kesimpulan tetap | S8, C10 |
| Asosiasi | Dua hal muncul bersamaan, tanpa bukti bahwa yang satu menyebabkan yang lain | S9, S11, C5 |
| Sebab-akibat (kausal) | Satu hal benar-benar menyebabkan yang lain. SHAP tidak bisa membuktikannya | S9, S11, S12, C5 |
| Pola komposisi | Porsi jenis pemanas selalu berjumlah 100%, jadi kalau satu porsi naik, porsi lain otomatis turun. Pola ini muncul dari aturan jumlah, bukan dari perilaku | S11 |
