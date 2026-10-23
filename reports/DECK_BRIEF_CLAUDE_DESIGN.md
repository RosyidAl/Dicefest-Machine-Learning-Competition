# Brief Pitch Deck untuk Claude Design (Final DICE FEST 2026, Tim BISMILLAH)

Dokumen ini adalah bahan masukan untuk Claude Design. Bagian A berisi cara kerja yang disarankan, Bagian B berisi
prompt utama yang tinggal ditempel, Bagian C berisi isi setiap slide, dan Bagian D berisi daftar periksa angka.
Semua angka sudah disamakan dengan writeup final (`BISMILLAH_Writeups.pdf`, sha 75b44b37).

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
6. **Latihan dengan pengatur waktu.** Total 10 menit; pembagian waktu per slide ada di Bagian C.

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

### Slide 1. Judul (30 detik)
- **Judul:** Dari Struktur Data ke Peta Kebijakan: Prediksi Trilema Energi Perkotaan dan Prioritas per Zona
- **Subjudul:** Tim BISMILLAH, DICE FEST 2026 Machine Learning
- **Angka besar:** 0,24844 (*leaderboard* private, peringkat 8 dari seluruh peserta) dan 0,24913 (publik)
- **Catatan pembicara:** Lima besaran energi diprediksi untuk sekitar 613 ribu kawasan permukiman, lalu model yang sama
  diterjemahkan menjadi peta prioritas untuk 330 zona: di mana *retrofit* didahulukan, di mana *heat pump* dapat segera
  dipasang, dan di mana jaringan listrik perlu diperkuat lebih dulu.

### Slide 2. Lima target, satu trilema (45 detik)
- **Judul:** Lima target mencerminkan tiga tuntutan transisi energi
- **Isi:** tabel kecil
  | Target | Aspek trilema |
  |---|---|
  | gas total, EUI (gas per m² lantai) | emisi dan biaya |
  | listrik total | keandalan jaringan |
  | porsi bangunan bersih | kualitas stok bangunan |
  | adopsi pemanas listrik | kemajuan transisi |
- **Pesan kecil:** metrik = rata-rata RMSE/σ per target; EUI menyumbang sekitar separuh total galat.
- **Gambar:** `fig1_targets.png` (opsional, kecil di bawah tabel)
- **Catatan pembicara:** Setiap target dinilai terpisah sehingga setiap target boleh ditangani dengan cara berbeda.
  Karena EUI menyumbang separuh galat, sebagian besar upaya pemodelan dicurahkan ke sana.

### Slide 3. Data dipahami sebelum model dilatih (1 menit)
- **Judul:** Sebelum melatih model, dua target sudah terjawab oleh struktur data
- **Isi (3 butir + 1 bukti):**
  - Gas total = EUI × luas lantai (`bld_dim_06`), selisih terbesar 7 × 10⁻¹⁶: gas dan EUI ditangani satu kelompok model.
  - Porsi bangunan bersih = 100 × `bld_dim_02` / `bld_dim_01`, tepat di seluruh 429 ribu baris latih: dihitung, tidak ditebak.
  - Adopsi pemanas listrik seragam dalam satu subzona: 92% kawasan uji sudah diketahui nilainya.
  - Data uji diambil acak per baris: kawasan uji di subzona baru 8,14% (harapan acak 8,17%), AUC validasi adversarial 0,4997.
- **Visual:** tiga kotak "identitas" dengan ikon sederhana atau tanpa ikon; tidak perlu gambar.
- **Catatan pembicara:** Temuan ini menentukan seluruh desain: satu target dihitung dengan rumus, satu target hampir
  seluruhnya disalin, dan dua target lain dimodelkan bersama.

### Slide 4. Skema validasi yang meniru data uji (45 detik)
- **Judul:** Skema validasi yang salah akan membuang fitur terbaik
- **Isi:**
  - Pembagian per zona membuat rata-rata target wilayah tampak merusak: galat EUI naik dari 0,6923 menjadi 0,8331.
  - Pembagian acak per baris, yang sesuai dengan data uji, justru menurunkan galat: 0,6721 menjadi 0,6699.
  - *Holdout set* 30% menduga perubahan skor *leaderboard* dengan akurat sehingga tidak ada keputusan yang diambil dari *leaderboard* publik.
- **Gambar:** `fig3_progress.png`
- **Catatan pembicara:** Untuk data spasial, pembagian per wilayah sering dianjurkan. Namun, 92% kawasan uji berada di
  subzona yang juga ada di data latih, sehingga skema per zona justru menyesatkan.

### Slide 5. Fitur domain (1 menit 15 detik)
- **Judul:** Fitur dibangun dari cara energi dipakai, bukan dari ukuran kawasan
- **Isi (4 butir):**
  - Ukuran rumah: luas lantai per meter, luas per sertifikat, meter per sertifikat, sisa porsi jenis pemanas.
  - Target per rumah tangga (gas per meter, per m², log, Poisson) dengan bobot yang menjaga galat pada skala asli.
  - Rata-rata target wilayah *out-of-fold*: subzona, zona, petak iklim, nomor subzona berdekatan.
  - Kawasan termirip (kNN pada stok bangunan): perbaikan terbesar dari satu kelompok fitur, galat EUI turun 0,0022.
- **Gambar:** `fig5_gains.png` (panel atas)
- **Catatan pembicara:** Gas total suatu kawasan terutama ditentukan jumlah rumah di dalamnya. Yang informatif adalah
  perilaku per rumah, jadi model memprediksi besaran per rumah tangga lalu dikalikan kembali. Semua fitur yang memakai
  target dihitung tanpa melihat *fold* yang sedang dievaluasi.

### Slide 6. Batas akurasi yang bisa dicapai (30 detik)
- **Judul:** Tiga perempat galat berasal dari variasi antarrumah tangga
- **Isi:** varians galat mengikuti a + b/M (M = jumlah meter); makin banyak rumah tangga, makin kecil galatnya.
  Implikasinya: perbaikan datang dari mengurangi galat acak model (*ensemble*), bukan dari model yang lebih besar.
- **Gambar:** `fig2_noise_floor.png`
- **Catatan pembicara:** Faktor seperti jumlah penghuni dan kebiasaan tidak tercatat dalam data. Bagian ini tidak dapat
  dijelaskan oleh model mana pun.

### Slide 7. Arsitektur multi-output (1 menit 15 detik)
- **Judul:** Tiga jalur untuk lima target, 41 model dengan pola galat berbeda
- **Isi:**
  - Jalur gas/EUI (28 model), jalur listrik (13 model), jalur rumus dan salinan (porsi bangunan bersih, adopsi).
  - XGBoost, LightGBM, MLP (PLR, TabM-mini, gas + listrik), dan NCA.
  - Penggabungan NNLS dengan bobot yang berubah menurut luas, jumlah meter, dan besar prediksi; lalu koreksi galat
    wilayah dan penyesuaian akhir (kawasan 2 sampai 4 meter, batas gas 999.765 kWh).
  - Reprodusibilitas: seluruh model dilatih ulang dari awal dan berkas submisi identik byte (sha256 aec745ab).
- **Gambar:** `fig_arch.png`
- **Catatan pembicara:** Model yang unggul di kawasan kecil tidak selalu unggul di kawasan besar, sehingga bobot
  penggabungan dibuat adaptif. Semua besaran setelah penggabungan ditentukan pada *holdout set*.

### Slide 8. Apa yang mendorong konsumsi (1 menit 30 detik)
- **Judul:** Ukuran rumah, periode bangunan, dan iklim menggerakkan konsumsi
- **Isi (angka besar):**
  - Rumah kecil: gas per m² sekitar **65%** lebih tinggi. Rumah besar: listrik per rumah sekitar **75%** lebih tinggi.
  - Periode `bld_era_15`: EUI sekitar **7%** lebih tinggi. Iklim menggeser EUI sekitar **15%**.
  - Adopsi pemanas listrik terkonsentrasi di kawasan yang belum tersambung jaringan gas; sosiodemografi hanya **15%**.
- **Gambar:** `pol_dependence.png` (utama) dan `pol_families.png` (kecil)
- **Catatan pembicara:** Model penjelas hanya memakai karakteristik kawasan yang terukur dan menjelaskan 75% variasi gas
  per meter, 56% variasi EUI, dan 49% variasi listrik per meter. Porsi 81% komposisi pemanas pada adopsi sebagian besar
  adalah identitas akuntansi, jadi dibaca sebagai deskripsi, bukan faktor pendorong. SHAP menunjukkan asosiasi, bukan
  sebab-akibat.

### Slide 9. Peta trilema 330 zona (1 menit 30 detik)
- **Judul:** Setiap zona membutuhkan urutan program yang berbeda
- **Angka besar:** potensi penghematan *retrofit* 7,4 TWh (5,2% konsumsi gas); elektrifikasi penuh menaikkan konsumsi
  listrik tahunan zona rata-rata 95%
- **Tabel kuadran:**
  | Kuadran | Zona | Porsi gas | Potensi *retrofit* | Kenaikan listrik tahunan | Langkah |
  |---|---|---|---|---|---|
  | Q1 intensitas tinggi, kenaikan tinggi | 118 | 47% | 3,65 TWh | +102% | *retrofit* dulu, lalu elektrifikasi bertahap |
  | Q2 intensitas tinggi, kenaikan relatif lebih rendah | 47 | 12% | 1,07 TWh | +90% | kandidat *heat pump* jangka pendek |
  | Q3 intensitas rendah, kenaikan tinggi | 47 | 15% | 0,86 TWh | +99% | penguatan gardu dan tarif fleksibel dulu |
  | Q4 intensitas rendah, kenaikan relatif lebih rendah | 118 | 27% | 1,81 TWh | +87% | program standar |
- **Gambar:** `pol_trilemma.png`
- **Catatan pembicara:** Angka kenaikan adalah energi tahunan dengan asumsi 80% gas untuk pemanasan dan COP 3, bukan
  beban puncak. Dengan COP 2,5 sampai 3,5 dan porsi pemanasan 70% sampai 90%, rata-ratanya 71% sampai 128%, tetapi
  peringkat zona dan keanggotaan kuadran tidak berubah.

### Slide 10. Rekomendasi dan penutup (1 menit 30 detik)
- **Judul:** Lima rekomendasi untuk pemerintah kota
- **Isi:**
  1. Urutkan program per kuadran, jangan diseragamkan. Q1 (REG_273, REG_270, REG_045) mendahulukan *retrofit*.
  2. Jadikan Q2 kandidat jangka pendek *heat pump*, setelah kesiapan gardu dikonfirmasi operator jaringan.
  3. Libatkan operator jaringan sejak awal di Q1 dan Q3 (misalnya REG_290, +121%): penguatan gardu, tarif berbasis
     waktu pemakaian, *heat pump* dengan penyimpan panas.
  4. Beri insentif khusus untuk kawasan yang sudah tersambung jaringan gas (porsi di atas 80%).
  5. Pisahkan target efisiensi (rumah kecil, bangunan lama) dari target emisi (zona dengan volume gas terbesar).
- **Baris penutup:** struktur data sebelum model; validasi yang meniru data uji; interpretasi yang bisa dieksekusi.
- **Catatan pembicara:** Tiga keterbatasan disebutkan singkat: SHAP adalah asosiasi, skenario listrik adalah energi
  tahunan bukan beban puncak, dan ambang kuadran berupa median. Karena itu peringkat antarzona lebih layak dipegang
  daripada nilai absolutnya.

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
