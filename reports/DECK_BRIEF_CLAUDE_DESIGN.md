# Brief Pitch Deck untuk Claude Design (Final DICE FEST 2026, Tim BISMILLAH)

Dokumen ini adalah bahan masukan untuk Claude Design. Bagian A berisi cara kerja yang disarankan, Bagian B berisi
prompt utama yang tinggal ditempel, Bagian C berisi isi setiap slide, dan Bagian D berisi daftar periksa angka.
Sumber angka: writeup final (`BISMILLAH_Writeups.pdf`, sha 75b44b37) dan `reports/deck_facts_extra.md` (skor
*leaderboard* dan Lampiran A notebook). Brief ini lolos `check_deck.py` (0 galat, 0 peringatan).

## A. Cara kerja yang disarankan

1. **Siapkan bahan sebelum membuka Claude Design.** Unggah tiga hal sekaligus: prompt utama (Bagian B), isi slide
   (Bagian C), dan gambar dari `reports/figures/` sesuai daftar di bawah. Writeup PDF boleh ikut diunggah sebagai
   rujukan, tetapi jangan dijadikan satu-satunya masukan karena 14 halaman terlalu panjang untuk diringkas tanpa arahan.
2. **Minta rancangan struktur dulu, bukan langsung jadi.** Perintah pertama: "Usulkan tata letak untuk setiap slide
   berdasarkan isi ini, belum perlu dibuat." Periksa usulannya, baru minta dibuat.
3. **Revisi per slide dengan perintah spesifik**, misalnya "Slide 8: perbesar gambar dependensi, kurangi teks menjadi
   tiga butir, tampilkan angka 65% dan 75% sebagai angka besar." Hindari perintah umum seperti "buat lebih bagus".
4. **Kunci angka.** Setelah draf jadi, minta: "Cocokkan setiap angka di semua slide dengan Bagian D; laporkan yang
   berbeda." Claude Design bisa saja membulatkan atau mengubah angka saat merapikan teks.
5. **Ekspor di akhir saja.** Untuk presentasi Zoom, PDF paling aman. Ekspor PPTX atau Canva hanya bila perlu disunting
   lagi, lalu periksa font dan posisi teks karena hasil ekspor PPTX bisa bergeser.
6. **Latihan dengan pengatur waktu.** Total 10 menit; pembagian waktu per slide ada di judul setiap slide di Bagian C.

Gambar yang perlu diunggah:

| Berkas | Dipakai di slide |
|---|---|
| `reports/figures/fig1_targets.png` | 2 |
| `reports/figures/fig3_progress.png` | 4 |
| `reports/figures/fig5_gains.png` | 5 |
| `reports/figures/fig2_noise_floor.png` | 6 |
| `reports/figures/fig_arch.png` | 7 |
| `reports/figures/pol_families.png` | 8 |
| `reports/figures/pol_dependence.png` | 8 |
| `reports/figures/pol_trilemma.png` | 9 |

Catatan: `fig_arch.png`, `pol_families.png`, `pol_dependence.png`, dan `pol_trilemma.png` beresolusi sekitar
800 sampai 1.200 piksel. Untuk tampilan layar penuh, sebaiknya dibuat ulang dengan resolusi lebih tinggi.

## B. Prompt utama (tempel ke Claude Design)

```
Buatkan pitch deck 16:9 berbahasa Indonesia untuk presentasi final kompetisi DICE FEST 2026 Machine Learning.
Presentasi berlangsung lewat Zoom: 10 menit paparan dan 10 menit tanya jawab. Penonton adalah dewan juri.

Bobot penilaian juri: rekomendasi kebijakan dari interpretabilitas SHAP 35%, rekayasa fitur domain dan
pra-pemrosesan 25%, slide dan tanya jawab 25%, arsitektur model dan kerapian kode 15%. Porsi slide dan waktu
mengikuti bobot ini, dengan bagian kebijakan paling menonjol.

Alur cerita: struktur data dipahami lebih dulu, lalu fitur yang meniru cara energi dipakai, lalu arsitektur
model, lalu peta prioritas kebijakan per zona. Pesan utamanya: memahami data lebih berharga daripada memperbesar
model, dan model yang sama dapat diterjemahkan menjadi daftar prioritas yang bisa langsung dipakai.

Gaya visual:
- Latar terang, teks abu-abu gelap (#2b2b2b), warna utama biru (#3b6ea5), aksen merah (#c0392b) hanya untuk
  angka atau temuan terpenting. Jangan memakai gradasi atau ikon dekoratif yang tidak membawa informasi.
- Satu pesan per slide. Judul slide berupa kalimat pernyataan singkat, bukan sekadar topik.
- Teks di layar maksimal sekitar 40 kata per slide. Angka kunci ditampilkan besar.
- Huruf sans-serif; judul minimal 28 pt, isi minimal 18 pt, keterangan gambar minimal 12 pt.
- Gambar yang diunggah dipakai apa adanya: jangan diubah, dipotong, atau digambar ulang. Gambar mengisi
  setidaknya separuh slide bila slide tersebut memakai gambar.
- Kaki slide: "Tim BISMILLAH | DICE FEST 2026" dan nomor slide.

Aturan isi:
- Pakai isi dan angka persis seperti pada dokumen isi slide. Jangan menambah data, klaim, atau angka baru.
- Bahasa Indonesia baku (EYD), kalimat pendek dan jelas. Istilah asing ditulis miring.
- Jangan memakai tanda pisah panjang (em-dash).
- Catatan pembicara dari dokumen isi dimasukkan ke speaker notes setiap slide.

Buat 10 slide utama dan 4 slide cadangan untuk sesi tanya jawab, sesuai dokumen isi slide.
Sebelum membuat slide, usulkan dulu tata letak setiap slide secara singkat.
```

## C. Isi per slide

Pembagian waktu mengikuti bobot rubrik: kebijakan (slide 8 sampai 10) 4 menit 20 detik, fitur dan pra-pemrosesan
(slide 3 sampai 6) 3 menit 25 detik, arsitektur (slide 7) 1 menit 5 detik, pembuka (slide 1 dan 2) 1 menit 10 detik.

### Slide 1. Judul (30 detik)
- **Judul:** Dari Struktur Data ke Peta Kebijakan: Prediksi Trilema Energi Perkotaan dan Prioritas per Zona
- **Subjudul:** Tim BISMILLAH, DICE FEST 2026 Machine Learning
- **Angka besar:** 0,24844 (*leaderboard* private, peringkat 8) dan 0,24913 (publik)
- **Catatan pembicara:** Kami memprediksi lima besaran energi untuk 613 ribu kawasan permukiman, lalu menerjemahkan
  model yang sama menjadi peta prioritas untuk 330 zona: di mana *retrofit* didahulukan, di mana *heat pump* dapat
  segera dipasang, dan di mana jaringan listrik perlu diperkuat lebih dulu.

### Slide 2. Lima target, satu trilema (40 detik)
- **Judul:** Lima target mencerminkan tiga tuntutan transisi energi
- **Isi:**

  | Target | Aspek trilema |
  |---|---|
  | gas total, EUI (gas per m² lantai) | emisi dan biaya |
  | listrik total | keandalan jaringan |
  | porsi bangunan bersih | kualitas stok bangunan |
  | adopsi pemanas listrik | kemajuan transisi |

- **Pesan kecil:** Metrik: rata-rata RMSE/σ per target. EUI menyumbang sekitar separuh total galat.
- **Gambar:** `fig1_targets.png` (opsional, kecil di bawah tabel)
- **Catatan pembicara:** Setiap target dinilai terpisah sehingga setiap target boleh ditangani dengan cara berbeda.
  Karena EUI menyumbang sekitar separuh galat, sebagian besar upaya pemodelan dicurahkan ke sana.

### Slide 3. Data dipahami sebelum model dilatih (1 menit)
- **Judul:** Dua target terjawab oleh struktur data sebelum model dilatih
- **Isi:**
  - Gas = EUI × luas lantai: satu kelompok model untuk keduanya.
  - Porsi bangunan bersih dihitung eksak dengan rumus.
  - Adopsi seragam per subzona: 92% kawasan uji sudah diketahui.
  - Data uji diambil acak per baris: AUC adversarial 0,4997.
- **Catatan pembicara:** Hasil kali EUI dan luas lantai (`bld_dim_06`) selalu sama dengan gas total, dengan selisih
  relatif terbesar 7 × 10⁻¹⁶. Porsi bangunan bersih selalu sama dengan 100 × `bld_dim_02` / `bld_dim_01`. Adopsi seragam
  dalam subzona karena subzona adalah unit sensus: 113 dari 132 kolom sensus juga seragam. Kawasan uji di subzona baru
  8,14%, sesuai harapan pengacakan per baris 8,17%. Satu target dihitung dengan rumus, satu hampir seluruhnya disalin,
  dan dua target lain dimodelkan bersama.

### Slide 4. Skema validasi yang meniru data uji (45 detik)
- **Judul:** Skema validasi yang keliru akan membuang fitur terbaik
- **Angka besar:** galat EUI dengan rata-rata target wilayah: per zona 0,8331, acak per baris 0,6699
- **Isi:** *Holdout set* 30% menduga perubahan skor *leaderboard* secara akurat.
- **Gambar:** `fig3_progress.png`
- **Catatan pembicara:** Untuk data spasial, pembagian per wilayah sering dianjurkan. Namun, 92% kawasan uji berada di
  subzona yang juga ada di data latih. Skema per zona membuat rata-rata target wilayah tampak merusak, padahal pada skema
  acak per baris yang sesuai dengan data uji, fitur yang sama menurunkan galat EUI dari 0,6721 menjadi 0,6699. Tidak ada
  keputusan yang diambil dari *leaderboard* publik.

### Slide 5. Fitur domain (1 menit 10 detik)
- **Judul:** Fitur dibangun dari cara energi dipakai, bukan ukuran kawasan
- **Isi:**
  - Ukuran rumah: luas lantai per meter, luas per sertifikat.
  - Target per rumah tangga dengan bobot skala asli.
  - Rata-rata target wilayah *out-of-fold*.
  - Kawasan termirip (kNN): galat EUI turun 0,0022.
- **Gambar:** `fig5_gains.png` (panel atas)
- **Catatan pembicara:** Gas total suatu kawasan terutama ditentukan jumlah rumah di dalamnya. Yang informatif adalah
  perilaku per rumah, jadi model memprediksi gas per meter, per m², atau logaritmanya, lalu dikalikan kembali. Rata-rata
  target dihitung pada subzona, zona, petak iklim, dan nomor subzona berdekatan, selalu tanpa melihat *fold* yang sedang
  dievaluasi. Kawasan termirip adalah perbaikan terbesar dari satu kelompok fitur.

### Slide 6. Batas akurasi yang bisa dicapai (30 detik)
- **Judul:** Tiga perempat galat berasal dari variasi antarrumah tangga
- **Isi:** Varians galat mengikuti a + b/M. Perbaikan datang dari *ensemble*, bukan dari model yang lebih besar.
- **Gambar:** `fig2_noise_floor.png`
- **Catatan pembicara:** M adalah jumlah meter. Makin banyak rumah tangga dalam satu kawasan, makin kecil galatnya.
  Jumlah penghuni dan kebiasaan tidak tercatat dalam data, jadi bagian ini tidak dapat dijelaskan oleh model mana pun.

### Slide 7. Arsitektur multi-output (1 menit 5 detik)
- **Judul:** Tiga jalur untuk lima target, 41 model dengan pola galat berbeda
- **Isi:**
  - Gas/EUI 28 model, listrik 13 model, dua target lain dengan rumus dan salinan.
  - XGBoost, LightGBM, MLP, NCA; bobot NNLS adaptif, lalu koreksi galat wilayah.
  - Dilatih ulang dari awal: berkas submisi identik byte (sha256 aec745ab).
- **Gambar:** `fig_arch.png`
- **Catatan pembicara:** Model yang unggul di kawasan kecil tidak selalu unggul di kawasan besar, sehingga bobot
  penggabungan berubah menurut luas, jumlah meter, dan besar prediksi. Setelah itu ada koreksi galat wilayah dan
  penyesuaian akhir untuk kawasan 2 sampai 4 meter serta batas gas 999.765 kWh. Semua besaran setelah penggabungan
  ditentukan pada *holdout set*.

### Slide 8. Apa yang mendorong konsumsi (1 menit 30 detik)
- **Judul:** Ukuran rumah, periode bangunan, dan iklim menggerakkan konsumsi
- **Angka besar:**
  - Rumah kecil: gas per m² +65%. Rumah besar: listrik per rumah +75%.
  - Periode `bld_era_15`: EUI +7%. Iklim menggeser EUI 15%.
  - Adopsi mengikuti sambungan jaringan gas; sosiodemografi hanya 15%.
- **Gambar:** `pol_dependence.png` (utama) dan `pol_families.png` (kecil)
- **Catatan pembicara:** Model penjelas hanya memakai karakteristik kawasan yang terukur dan menjelaskan 75% variasi gas
  per meter, 56% variasi EUI, dan 49% variasi listrik per meter. Porsi 81% komposisi pemanas pada adopsi sebagian besar
  adalah identitas akuntansi, jadi dibaca sebagai deskripsi, bukan faktor pendorong. SHAP menunjukkan asosiasi, bukan
  sebab-akibat.

### Slide 9. Peta trilema 330 zona (1 menit 30 detik)
- **Judul:** Setiap zona membutuhkan urutan program yang berbeda
- **Angka besar:** potensi *retrofit* 7,4 TWh (5,2% gas); elektrifikasi penuh menaikkan konsumsi listrik tahunan zona
  rata-rata 95%
- **Tabel kuadran:**

  | Kuadran | Zona (porsi gas) | Potensi *retrofit* | Kenaikan listrik | Langkah |
  |---|---|---|---|---|
  | Q1 intensitas tinggi, kenaikan tinggi | 118 (47%) | 3,65 TWh | +102% | *retrofit* dulu |
  | Q2 intensitas tinggi, kenaikan lebih rendah | 47 (12%) | 1,07 TWh | +90% | kandidat *heat pump* |
  | Q3 intensitas rendah, kenaikan tinggi | 47 (15%) | 0,86 TWh | +99% | perkuat gardu dulu |
  | Q4 intensitas rendah, kenaikan lebih rendah | 118 (27%) | 1,81 TWh | +87% | program standar |

- **Gambar:** `pol_trilemma.png`
- **Catatan pembicara:** Kenaikan adalah energi tahunan dengan asumsi 80% gas untuk pemanasan dan COP 3, bukan beban
  puncak. Dengan COP 2,5 sampai 3,5 dan porsi pemanasan 70% sampai 90%, rata-ratanya 71% sampai 128%, tetapi peringkat
  zona dan keanggotaan kuadran tidak berubah. Q1 memuat hampir separuh konsumsi gas dan separuh potensi *retrofit*.

### Slide 10. Rekomendasi (1 menit 20 detik)
- **Judul:** Lima langkah untuk pemerintah kota
- **Isi:**
  1. Urutkan program per kuadran; Q1 mendahulukan *retrofit*.
  2. Q2: kandidat *heat pump* jangka pendek.
  3. Libatkan operator jaringan sejak awal di Q1 dan Q3.
  4. Insentif khusus untuk kawasan berjaringan gas.
  5. Pisahkan target efisiensi dan target emisi.
- **Catatan pembicara:** Kandidat pertama *retrofit* adalah REG_273, REG_270, dan REG_045. Di Q2, kesiapan gardu tetap
  dikonfirmasi dengan operator jaringan sebelum subsidi dibuka. Zona seperti REG_290 (+121%) perlu penguatan gardu,
  tarif berbasis waktu pemakaian, dan *heat pump* dengan penyimpan panas. Kawasan dengan porsi gas jaringan di atas 80%
  perlu bantuan biaya pemasangan. Program keterjangkauan menyasar rumah kecil dan bangunan lama, program emisi menyasar
  zona dengan volume gas terbesar. Keterbatasan: SHAP adalah asosiasi, skenario listrik adalah energi tahunan, dan
  ambang kuadran berupa median, sehingga peringkat antarzona lebih layak dipegang daripada nilai absolutnya.

### Slide cadangan (untuk tanya jawab)
- **C1. Perbandingan skema validasi:** Tabel 1 writeup (acak per baris dan per zona, dengan dan tanpa rata-rata target
  wilayah; gas, EUI, listrik).
- **C2. Komposisi ensemble:** Tabel 2 writeup (jumlah model, galat terbaik, dan bobot per kelompok).
- **C3. Sensitivitas skenario listrik:** Tabel 4 writeup (porsi pemanasan 70/80/90% × COP 2,5/3/3,5: +100/+83/+71,
  +114/+95/+81, +128/+107/+92); efisiensi *boiler* 88% menurunkan semua nilai sekitar 12%.
- **C4. Model adopsi subzona baru:** N = jumlah rumah tangga subzona dari pecahan komposisi pemanas, K = rumah tangga di
  luar kategori tercatat; RMSE 1,03 poin persentase (0,143 ternormalisasi) pada 8% kawasan uji di subzona baru.

## D. Daftar periksa angka

| Angka | Nilai |
|---|---|
| Skor *leaderboard* private / publik | 0,24844 (peringkat 8) / 0,24913 |
| Data | 429 ribu kawasan latih, 184 ribu kawasan uji, 201 kolom fitur, 330 zona |
| Identitas gas = EUI × luas | selisih relatif terbesar 7 × 10⁻¹⁶ |
| Kawasan uji dengan adopsi diketahui | 92% |
| Subzona baru di data uji | 8,14% (harapan acak 8,17%) |
| AUC validasi adversarial | 0,4997 |
| Galat EUI per zona vs acak (dengan rata-rata wilayah) | 0,8331 vs 0,6699 |
| Perbaikan fitur kawasan termirip | EUI -0,0022 |
| Jumlah model | 41 (28 gas/EUI, 13 listrik) |
| Batas gas | 999.765 kWh |
| R² model penjelas | 75% gas per meter, 56% EUI, 49% listrik per meter |
| Pendorong | rumah kecil +65% gas/m², rumah besar +75% listrik/rumah, `bld_era_15` +7% EUI, iklim 15%, sosiodemografi 15% pada adopsi |
| Potensi *retrofit* | 7,4 TWh dari 141,4 TWh (5,2%) |
| Kenaikan konsumsi listrik tahunan | rata-rata 95%, antarzona 68% sampai 121%, sensitivitas 71% sampai 128% |
| Kuadran | Q1 118 zona, 47% gas, 3,65 TWh, +102%; Q2 47, 12%, 1,07 TWh, +90%; Q3 47, 15%, 0,86 TWh, +99%; Q4 118, 27%, 1,81 TWh, +87% |
