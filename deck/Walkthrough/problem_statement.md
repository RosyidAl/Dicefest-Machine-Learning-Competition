# Problem Statement: Trilema Transisi Energi Perkotaan

Dokumen ini membatasi diri pada studi kasus DICE FEST 2026 Machine Learning: masalah apa yang diajukan, apa yang
harus dicapai, dan batasan apa yang berlaku. Cara penyelesaian tidak dibahas di sini; uraiannya ada di
`walkthrough_tahapan.md` dan `solusi_walkthrough.md`.

Sumber: `MASTER_PROMPT.md` bagian 2 (fakta kompetisi), `reports/WRITEUP_DRAFT_v21.md` bab 1, dan rubrik babak final
yang disampaikan panitia.

## 1. Latar belakang

Sebuah kota hendak mengurangi ketergantungan pada pemanas berbahan gas. Peralihan ini menghadapi tiga tuntutan yang
saling menarik, yang disebut trilema transisi energi:

| Tuntutan | Isi | Risiko bila diabaikan |
|---|---|---|
| Emisi | konsumsi gas harus turun | target dekarbonisasi tidak tercapai |
| Keandalan jaringan | jaringan listrik harus tetap mampu menanggung beban ketika pemanas beralih ke listrik | gardu kelebihan beban |
| Keterjangkauan | biaya energi rumah tangga tidak boleh melonjak | beban biaya jatuh pada rumah tangga yang paling rentan |

Mengejar satu tuntutan tanpa memperhitungkan dua lainnya dapat menggagalkan program. Elektrifikasi yang cepat
menurunkan emisi tetapi menaikkan beban listrik; perbaikan bangunan (*retrofit*, renovasi untuk menurunkan kebutuhan
energi bangunan) menurunkan beban tetapi memerlukan biaya awal. Pemerintah kota karena itu memerlukan informasi
tentang profil energi setiap kawasan agar urutan program dapat disesuaikan per wilayah.

## 2. Data yang tersedia

- Satu baris adalah agregat satu kawasan permukiman, bukan satu rumah tangga.
- Data latih 429.303 kawasan (209 kolom) dan data uji 183.987 kawasan (204 kolom, tanpa target); total 613.290.
- 201 fitur numerik dengan nama yang disamarkan, terbagi dalam lima kelompok:

| Awalan | Jumlah | Isi menurut panitia |
|---|---|---|
| `bld_dim_*` | 43 | dimensi dan geometri bangunan |
| `bld_era_*` | 15 | sebaran era konstruksi (pra-1919 sampai pasca-1999) |
| `hvac_dist_*` | 10 | bauran sistem pemanas |
| `env_climate_*` | 6 | beban termal iklim mikro |
| `socio_demog_*` | 127 | okupansi, kepadatan, aktivitas, dan status kepemilikan rumah tangga |

- Dua kode lokasi: `zone_region_id` (330 zona) dan `subzone_id` (subzona yang lebih kecil).
- Data eksternal tidak boleh dipakai, sehingga semua informasi harus digali dari kolom di atas.

## 3. Masalah yang harus dijawab

Studi kasus ini memuat dua masalah yang berurutan.

### 3.1 Masalah prediksi

Untuk setiap kawasan di data uji, lima besaran energi harus diperkirakan sekaligus (*multi-output regression*,
regresi dengan beberapa keluaran). Kelima target memotret sisi trilema yang berbeda:

| Target | Arti | Satuan | Sisi trilema |
|---|---|---|---|
| `target_macro_gas_kwh` | konsumsi gas tahunan kawasan | kWh | emisi |
| `target_macro_elec_kwh` | konsumsi listrik tahunan kawasan | kWh | keandalan jaringan |
| `target_eui_gas_kwh_per_sqm` | intensitas gas per m² lantai (*Energy Use Intensity*) | kWh/m² | emisi dan biaya |
| `target_clean_building_ratio` | porsi bangunan bersih | % | kualitas stok bangunan |
| `target_electric_heating_adoption` | porsi rumah tangga berpemanas listrik | % | kemajuan transisi |

### 3.2 Masalah kebijakan

Prediksi saja belum menjawab kebutuhan pemerintah kota. Model harus diterjemahkan menjadi keputusan, dengan
pertanyaan berikut:
1. Karakteristik kawasan apa yang berasosiasi dengan konsumsi gas, konsumsi listrik, dan adopsi pemanas listrik?
2. Zona mana yang sebaiknya mendahulukan *retrofit*?
3. Zona mana yang dapat segera beralih ke pompa panas listrik (*heat pump*)?
4. Zona mana yang jaringan listriknya perlu diperkuat sebelum elektrifikasi?

Rubrik babak final mensyaratkan rekomendasi ini diturunkan dari interpretabilitas model (SHAP atau *feature
importance*, tingkat pengaruh setiap fitur terhadap prediksi).

## 4. Tantangan yang melekat pada data

Bagian ini mencatat kesulitan yang muncul dari bentuk data, bukan cara mengatasinya.

| Tantangan | Akibat bila tidak ditangani |
|---|---|
| Nama fitur disamarkan | arti kolom (misalnya luas lantai atau jumlah meter) harus ditemukan lewat analisis; fitur domain tidak dapat langsung dibuat |
| Mekanisme pembagian data latih dan uji tidak dinyatakan | skema validasi dapat menyimpang dari data uji, sehingga pemilihan fitur dan model salah arah |
| Skala target sangat berbeda (puluhan sampai ratusan ribu kWh, puluhan kWh/m², 0 sampai 100%) | perlu penanganan per target; RMSE di skala asli didominasi kawasan bernilai besar |
| Sebaran target berekor panjang: 1% baris teratas menyumbang 14% sampai 35% jumlah kuadrat deviasi | sejumlah kecil kawasan besar menentukan sebagian besar galat |
| Perilaku rumah tangga (jumlah penghuni, kebiasaan, termostat) tidak tercatat | ada bagian galat yang tidak dapat dijelaskan oleh fitur apa pun |
| Data bersifat observasional | hasil interpretabilitas menunjukkan asosiasi, bukan sebab-akibat |
| Data tidak memuat harga energi atau biaya investasi | tuntutan keterjangkauan hanya dapat didekati secara tidak langsung |

## 5. Tujuan yang ingin dicapai

| No. | Tujuan | Ukuran keberhasilan |
|---|---|---|
| T1 | Prediksi lima target seakurat mungkin | Column-Wise NRMSE serendah mungkin di *leaderboard* (papan peringkat) privat |
| T2 | Menjelaskan faktor yang berasosiasi dengan konsumsi dan adopsi | arah dan besar pengaruh setiap kelompok fitur, dilaporkan dengan batasannya |
| T3 | Menyusun prioritas kebijakan per zona | setiap zona dari 330 zona memperoleh urutan program yang dapat ditelusuri ke prediksi model |
| T4 | Reprodusibilitas | notebook menghasilkan berkas submisi yang identik saat diaudit juri |

### 5.1 Metrik

Skor = (1/5) × Σ RMSE_k / σ_k untuk lima target, dengan σ beku dari panitia:

| Target | σ |
|---|---|
| gas | 151.750,04 |
| listrik | 41.725,37 |
| EUI | 40,65 |
| porsi bangunan bersih | 12,49 |
| adopsi pemanas listrik | 7,18 |

Konsekuensi langsung dari rumus ini:
- Tebakan rata-rata yang sama untuk semua kawasan memberi sekitar 1,0 per target; angka ini menjadi titik acuan.
- Metrik dihitung terpisah per kolom, sehingga setiap target boleh ditangani dengan cara yang berbeda.
- Evaluasi dilakukan di skala asli. Transformasi target boleh dipakai saat pelatihan, tetapi galat dihitung setelah
  dikembalikan ke satuan semula.

### 5.2 Tahapan penilaian

| Tahap | Penentu | Bobot |
|---|---|---|
| Penyisihan (Top 10) | skor *leaderboard* privat (70% data uji) | 95% |
| Final: rekomendasi kebijakan dari interpretabilitas | SHAP atau *feature importance* | 35 poin |
| Final: rekayasa fitur domain dan pra-pemrosesan | kebaruan variabel spasial dan fisik, validitas pemrosesan | 25 poin |
| Final: *slide* dan tanya jawab | kejelasan visual, paparan 10 menit, jawaban | 25 poin |
| Final: arsitektur model dan kerapian kode | arsitektur *multi-output* yang rapi, notebook yang mudah dibaca | 15 poin |

Skor *leaderboard* hanya menentukan kelolosan ke final. Di babak final, 60 dari 100 poin berada pada interpretasi,
kebijakan, dan fitur, sehingga model harus dapat dijelaskan, tidak cukup hanya akurat.

## 6. Batasan

### 6.1 Aturan kompetisi
- Hanya data resmi (`train.csv`, `test.csv`); data eksternal, *scraping* sensus, dan data geografis tambahan dilarang.
- Nilai prediksi wajib keluar dari kode *open-source* yang dijalankan lokal; API LLM tertutup atau berbayar dan
  pelabelan manual dilarang.
- *Seed* dikunci di semua pustaka dan pembagian data.
- Submisi 183.987 baris dengan urutan `id` sama dengan `sample_submission.csv`; target persen berada di rentang 0
  sampai 100, target lain tidak negatif.
- Lima submisi per hari, dua submisi dipilih sebagai final. *Leaderboard* publik memakai 30% data uji dan privat 70%.
- Notebook `.ipynb` dan *writeup* PDF diserahkan paling lambat 23 Oktober 2026 pukul 23.59 WIB; presentasi final
  30 Oktober 2026 lewat Zoom (10 menit paparan, 10 menit tanya jawab).

### 6.2 Di luar lingkup
- Klaim sebab-akibat. Data observasional hanya mendukung pernyataan asosiasi.
- Beban puncak jaringan. Target listrik berupa energi tahunan, bukan daya puncak per jam.
- Perhitungan biaya dalam rupiah atau mata uang lain. Data tidak memuat harga energi maupun biaya investasi.
- Prediksi tingkat rumah tangga. Unit terkecil yang tersedia adalah kawasan.

## 7. Rumusan masalah

1. Bagaimana memperkirakan lima besaran energi untuk 183.987 kawasan permukiman dari 201 fitur anonim dengan
   Column-Wise NRMSE serendah mungkin, tanpa data eksternal, dan dengan hasil yang dapat direproduksi?
2. Karakteristik kawasan apa yang berasosiasi dengan konsumsi gas, konsumsi listrik, dan adopsi pemanas listrik?
3. Bagaimana hasil tersebut diterjemahkan menjadi urutan program yang berbeda untuk 330 zona, dengan
   mempertimbangkan ketiga tuntutan trilema secara bersamaan?
