---
name: pitch-deck-kompetisi
description: Menyusun pitch deck presentasi final kompetisi data science, machine learning, atau hackathon dalam bahasa Indonesia, dari bahan riset (writeup, notebook, gambar, angka hasil) menjadi alur cerita per slide, sistem desain, catatan pembicara, pembagian waktu sesuai rubrik juri, bank tanya jawab, dan pemeriksaan angka otomatis, lalu diserahkan sebagai brief siap tempel untuk Claude Design atau spesifikasi untuk PPTX/Canva. Gunakan skill ini setiap kali pengguna ingin membuat, menyusun ulang, merapikan, atau mengecek slide, PPT, deck, atau bahan presentasi untuk lomba, juri, babak final, pitching hasil model, atau Claude Design, termasuk bila pengguna hanya berkata "bikin slide final", "susun PPT buat juri", atau "rapikan deck ini", walaupun kata "pitch deck" tidak disebut. Also use it for English requests about competition final presentations or jury pitch decks for ML/data projects.
---

# Pitch Deck Final Kompetisi

Juri babak final biasanya hanya punya 10 sampai 15 menit untuk memutuskan tiga hal: apakah tim memahami masalahnya,
apakah solusinya benar dan jujur, dan apakah hasilnya berguna. Deck yang kuat menjawab ketiganya dengan bukti yang
mudah dilihat, bukan dengan banyak teks. Skill ini mengubah bahan riset tim menjadi deck seperti itu.

## Alur kerja

Ikuti urutan ini. Setiap langkah menghasilkan bahan untuk langkah berikutnya, dan melompati langkah awal biasanya
berakhir dengan slide yang rapi tetapi isinya keliru.

### 1. Kumpulkan bahan dan batasan
Cari atau tanyakan sekali saja (lalu pakai asumsi yang dinyatakan bila tidak ada jawaban):
- rubrik penilaian beserta bobotnya;
- durasi paparan dan tanya jawab, format (daring atau luring), batas jumlah slide;
- sumber kebenaran angka, biasanya writeup final atau keluaran notebook;
- gambar yang sudah ada (nama berkas dan resolusinya);
- alat yang akan dipakai untuk membuat slide (Claude Design, PowerPoint, Canva, Google Slides).

Asumsi bawaan bila tidak diketahui: 10 menit paparan, 10 slide utama ditambah 3 sampai 4 slide cadangan, 16:9.

### 2. Buat lembar fakta
Tulis setiap angka dan klaim yang mungkin muncul di slide beserta sumbernya (bab writeup atau sel notebook). Slide
hanya boleh memuat angka yang ada di lembar fakta. Juri sering menguji satu angka, dan angka yang berbeda antara slide,
writeup, dan notebook langsung menurunkan kepercayaan pada seluruh pekerjaan. Lembar fakta ini juga dipakai skrip
pemeriksa di langkah 8.

### 3. Petakan rubrik ke slide dan waktu
Bagi slide dan waktu sebanding dengan bobot rubrik. Kriteria berbobot terbesar harus mendapat slide paling banyak dan
posisi yang tidak terburu-buru (jangan diletakkan di menit terakhir). Tulis pembagiannya sebagai tabel kecil, misalnya:

| Kriteria (bobot) | Slide | Waktu |
|---|---|---|
| kebijakan dari interpretabilitas (35) | 8, 9, 10 | 4 menit 30 detik |
| fitur dan pra-pemrosesan (25) | 3, 4, 5, 6 | 3 menit |
| arsitektur dan kode (15) | 7 | 1 menit 15 detik |
| pembuka dan penutup | 1, 2 | 1 menit 15 detik |

### 4. Tulis alur cerita sebagai judul-kalimat
Susun dulu daftar judul slide berupa kalimat pernyataan (*action title*), misalnya "Tiga perempat galat berasal dari
variasi antarrumah tangga", bukan "Analisis galat". Uji sederhananya: bila juri hanya membaca judul, ceritanya tetap
utuh. Alur yang biasanya berhasil untuk kompetisi ML: masalah, wawasan data, skema validasi, fitur, arsitektur, hasil,
interpretasi, rekomendasi, penutup. Pola tata letak setiap jenis slide ada di `references/slide-patterns.md`.

### 5. Spesifikasikan setiap slide
Untuk setiap slide tulis: judul-kalimat; isi di layar (paling banyak sekitar 40 kata dan 4 butir); visual (gambar,
tabel ringkas, atau angka besar); catatan pembicara 60 sampai 120 kata yang ditulis untuk diucapkan; durasi; dan sumber
angkanya. Satu slide membawa satu pesan. Bila sebuah slide butuh dua pesan, pecah menjadi dua slide atau pindahkan
salah satunya ke catatan pembicara.

### 6. Tetapkan sistem desain
- Rasio 16:9, latar terang, teks abu-abu gelap.
- Ambil warna utama dari gambar yang sudah ada agar slide dan grafik terlihat satu keluarga; satu warna aksen hanya
  untuk angka atau temuan terpenting.
- Huruf sans-serif; judul minimal 28 pt, isi minimal 18 pt, keterangan gambar minimal 12 pt, karena presentasi daring
  sering ditonton di layar kecil.
- Gambar mengisi setidaknya separuh slide yang memakainya. Gambar yang lebarnya di bawah sekitar 1.600 piksel akan
  buram bila dipasang besar; sarankan untuk dirender ulang daripada diperbesar.
- Kaki slide: nama tim, nama acara, nomor slide.

### 7. Siapkan bank tanya jawab
Tulis 8 sampai 12 pertanyaan yang paling mungkin diajukan juri beserta jawaban 2 sampai 3 kalimat dan slide cadangan
yang menjadi rujukannya. Daftar pertanyaan umum dan pola jawabannya ada di `references/qa-bank.md`. Tanya jawab sering
berbobot sama besar dengan slide itu sendiri, jadi bagian ini tidak boleh dianggap tambahan.

### 8. Periksa secara otomatis
Jalankan pemeriksa pada brief atau berkas PPTX:

```bash
python scripts/check_deck.py BRIEF.md --facts FAKTA.md --minutes 10
python scripts/check_deck.py deck.pptx --facts FAKTA.md --minutes 10   # perlu python-pptx
```

Skrip melaporkan tanda pisah panjang (em-dash), slide yang terlalu padat, total durasi, kata "kami"/"saya" di teks
layar, frasa generik seperti "revolusioner" atau "di era digital", dan angka yang tidak ada di lembar fakta. Lembar
fakta boleh lebih dari satu berkas (`--facts` diulang), misalnya writeup ditambah catatan skor *leaderboard*. Format
brief yang dibaca skrip mengikuti `assets/brief-template.md`. Perbaiki lalu jalankan lagi sampai bersih. Peringatan angka perlu
dibaca dengan akal sehat: nomor slide atau tahun boleh diabaikan, tetapi angka hasil yang tidak ada di lembar fakta
harus diperbaiki atau ditambahkan sumbernya.

### 9. Serahkan sesuai alat
- **Claude Design:** tulis brief dengan templat `assets/brief-template.md` (prompt utama, isi per slide, daftar gambar,
  daftar periksa angka), lalu ikuti `references/claude-design.md` untuk cara mengunggah, merevisi, dan mengekspor.
- **PPTX:** bila skill `pptx` tersedia, pakai skill itu untuk membangun berkas dari spesifikasi slide dan sistem desain
  di atas, lalu jalankan pemeriksa pada hasilnya.
- **Canva atau Google Slides:** serahkan brief yang sama; prompt utama dapat dipakai sebagai panduan desain manual.

## Gaya bahasa

Teks slide dibaca juri dalam beberapa detik, sedangkan catatan pembicara diucapkan. Keduanya ditulis berbeda.

- Bahasa Indonesia baku sesuai EYD V. Bila skill `indonesia-eyd-writing-skill` tersedia, gunakan untuk memeriksa
  ejaan dan tanda baca.
- Teks di layar bersifat impersonal (tanpa "kami" atau "saya") dan berupa frasa padat. Catatan pembicara boleh memakai
  "kami" karena diucapkan oleh tim, kecuali pengguna meminta gaya impersonal penuh.
- Istilah asing yang lazim dibiarkan dalam bahasa Inggris, dicetak miring, dan diberi penjelasan singkat dalam kurung
  pada kemunculan pertama, misalnya *holdout set* (data latih yang disisihkan untuk evaluasi internal).
- Jangan memakai tanda pisah panjang (em-dash). Ganti dengan koma, titik dua, atau kalimat baru. Tanda ini jarang
  dipakai dalam tulisan Indonesia dan sering dianggap ciri teks buatan mesin.
- Hindari ciri tulisan generik: pembuka seperti "Di era digital ini", kata seperti "revolusioner", "holistik",
  "sinergi", "inovatif" tanpa bukti, deretan tiga kata sifat, pertanyaan retoris berulang, emoji, dan cetak tebal
  berlebihan. Ganti dengan angka konkret, nama wilayah atau variabel yang spesifik, dan kata kerja yang jelas.
- Angka di teks memakai format Indonesia (koma desimal, titik ribuan) dan pembulatan yang sama dengan sumbernya.

## Prinsip isi untuk kompetisi ML

- **Wawasan data sebelum model.** Juri lebih terkesan oleh temuan struktur data yang mengubah desain solusi daripada
  oleh daftar model. Tampilkan temuan itu lebih awal.
- **Validasi yang jujur.** Jelaskan mengapa skema validasi meniru data uji dan bagaimana kebocoran dicegah; ini jawaban
  atas pertanyaan juri yang paling sering muncul.
- **Arsitektur dalam satu gambar.** Diagram alur satu halaman lebih efektif daripada tabel hiperparameter. Rincian
  masuk slide cadangan.
- **Interpretasi menjadi tindakan.** Hasil SHAP atau interpretasi lain diterjemahkan menjadi rekomendasi yang menyebut
  siapa, di mana, dan dalam urutan apa. Nyatakan bahwa interpretasi menunjukkan asosiasi, bukan sebab-akibat, dan sebut
  asumsi skenario secara eksplisit.
- **Hasil yang bisa dicek.** Sebut skor, posisi, perbedaan publik dan private bila ada, serta bukti reprodusibilitas
  (seed, hash berkas submisi).

## Rujukan

| Berkas | Kapan dibaca |
|---|---|
| `references/slide-patterns.md` | saat memilih tata letak setiap jenis slide (langkah 4 sampai 6) |
| `references/claude-design.md` | saat keluaran akan dibuat di Claude Design (langkah 9) |
| `references/qa-bank.md` | saat menyiapkan tanya jawab dan slide cadangan (langkah 7) |
| `references/case-dicefest-2026.md` | contoh lengkap satu deck final (DICE FEST 2026, tim BISMILLAH) sebagai acuan mutu |
| `assets/brief-template.md` | templat brief untuk Claude Design atau alat lain |
| `scripts/check_deck.py` | pemeriksa otomatis (langkah 8) |
