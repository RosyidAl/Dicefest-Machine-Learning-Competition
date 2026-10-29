# Brief Pitch Deck: <NAMA ACARA>, Tim <NAMA TIM>

Bahan masukan untuk alat pembuat slide (Claude Design, PowerPoint, Canva). Bagian A berisi cara kerja, Bagian B
prompt utama, Bagian C isi setiap slide, dan Bagian D daftar periksa angka. Sumber angka: <writeup/notebook final>.

## A. Cara kerja
1. Unggah prompt utama, isi slide, dan gambar dalam satu pesan.
2. Minta usulan tata letak setiap slide sebelum slide dibuat.
3. Revisi per slide dengan perintah spesifik.
4. Minta pencocokan semua angka dengan Bagian D.
5. Ekspor di akhir (PDF untuk presentasi daring).

| Berkas gambar | Dipakai di slide | Lebar (piksel) |
|---|---|---|
| <path/gambar.png> | <n> | <lebar> |

## B. Prompt utama

```
Buatkan pitch deck 16:9 berbahasa Indonesia untuk <acara>. Presentasi <daring/luring>: <X> menit paparan dan
<Y> menit tanya jawab. Penonton: <juri/investor/...>.

Bobot penilaian: <kriteria 1> <bobot>%, <kriteria 2> <bobot>%, ... Porsi slide dan waktu mengikuti bobot ini.

Alur cerita: <tiga sampai lima tahap>. Pesan utama: <satu kalimat>.

Gaya visual:
- Latar terang, teks <#kode>, warna utama <#kode>, aksen <#kode> hanya untuk angka atau temuan terpenting.
- Satu pesan per slide; judul slide berupa kalimat pernyataan singkat.
- Teks di layar maksimal sekitar 40 kata per slide; angka kunci ditampilkan besar.
- Gambar yang diunggah dipakai apa adanya dan mengisi setidaknya separuh slide yang memakainya.
- Kaki slide: "<tim> | <acara>" dan nomor slide.

Aturan isi:
- Pakai isi dan angka persis seperti pada dokumen isi slide; jangan menambah data atau klaim baru.
- Bahasa Indonesia baku (EYD), istilah asing dicetak miring, tanpa tanda pisah panjang (em-dash).
- Catatan pembicara dimasukkan ke speaker notes setiap slide.

Buat <N> slide utama dan <M> slide cadangan. Usulkan dulu tata letak setiap slide secara singkat.
```

## C. Isi per slide

### Slide 1. <Judul pendek> (<durasi, misalnya 30 detik>)
- **Judul:** <kalimat pernyataan>
- **Isi:** <maksimal 4 butir atau satu tabel kecil>
- **Angka besar:** <opsional>
- **Gambar:** <berkas> (opsional)
- **Catatan pembicara:** <60 sampai 120 kata, ditulis untuk diucapkan>

<ulangi untuk setiap slide>

### Slide cadangan (untuk tanya jawab)
- **C1. <judul>:** <isi>

## D. Daftar periksa angka

| Angka | Nilai | Sumber |
|---|---|---|
| <nama> | <nilai> | <bab writeup atau sel notebook> |
