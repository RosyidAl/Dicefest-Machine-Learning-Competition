# Skala prioritas persiapan presentasi final (30 Oktober 2026)

Dasar: bobot juri (kebijakan dan SHAP 35, rekayasa fitur 25, slide dan tanya jawab 25, arsitektur model dan kode 15)
dan kondisi bahwa tidak semua materi teknis perlu dikuasai sampai ke detail. Prinsipnya: sedikit hal dipahami dalam,
sisanya dihafal sebagai kalimat tetap. Pembagian waktu belajar di bawah adalah saran, bukan hasil pengukuran.

## Ringkasan skala

| Urutan | Bagian | Bobot juri | Saran porsi waktu belajar | Tingkat penguasaan |
|---|---|---|---|---|
| P1 | Rekomendasi kebijakan dan SHAP (slide 8 sampai 11) | 35 | 40% | paham alasan, bukan hanya hafal |
| P2 | Slide, waktu, dan tanya jawab (semua slide, C1 sampai C12) | 25 | 25% | latihan lisan |
| P3 | Rekayasa fitur dan pra-pemrosesan (slide 3 sampai 5, C12) | 25 | 20% | tiga kalimat per slide |
| P4 | Arsitektur model dan notebook (slide 6, 7) | 15 | 10% | satu kalimat dan peta bagian notebook |
| Lewati | Detail CV, bobot blending, pseudo-label, skenario COP selain angka utama | 0 langsung | 5% (hanya tahu letaknya) | tidak dipelajari |

## P1. Rekomendasi kebijakan dan SHAP (35 poin)

Yang dipahami:
- Satu rantai di tiap rekomendasi (R1, R2, R3): temuan model, angka pendukung, langkah yang disarankan, pelaksana.
- SHAP adalah kontribusi fitur terhadap prediksi model, bukan bukti sebab-akibat. Contoh: kontribusi SHAP +75% pada
  listrik per rumah untuk rumah besar dibanding kecil. Jawaban bila ditekan: "ini kontribusi dalam model, bukan klaim
  kausal".
- R2 (penilaian gardu) menyasar Q1 dan Q3, bukan Q1 dan Q2.
- Potensi hemat 7,39 TWh disebut "indikator" dengan bias dua arah, bukan batas atas.
- Frasa tetap: "temuan model mendukung rekomendasi ini".

Yang dihafal sebagai angka (semua sudah terverifikasi di `laporan_v5.md`):
- Slide 9: +65% dan +34% antarkuartil.
- Slide 10: REG_263, REG_045, REG_270 dengan tambahan 757, 442, 434 GWh (COP 3); REG_290 turun dari peringkat 1 ke 47
  pada COP per iklim.
- Slide 11: adopsi 9,46% (gas jaringan 60 sampai 80%) dibanding 2,73% (di atas 80%); contoh Q2 REG_033.

## P2. Slide, waktu, dan tanya jawab (25 poin)

- Durasi 9:30 dengan pembagian 60/150/90/210/60 detik. Latih dengan pengatur waktu, satu kali penuh minimal.
- Latih transisi antar slide dan siapa yang memegang laptop.
- Putuskan nama pembicara hari ini (masih `[PERLU DIISI]`). Pembagian dari audit: anchor (slide 1, 2, 14),
  teknis (slide 3 sampai 7), kebijakan (slide 8 sampai 13).
- Cadangan C1 sampai C12 tidak dihafal. Cukup tahu isinya dan nomornya.
- Kalimat jujur untuk hal di luar yang diuji: "Itu di luar yang kami uji. Yang kami buktikan adalah X."

Pertanyaan juri yang paling mungkin (rinci di bagian "Pertanyaan juri" `laporan_v5.md`):

| Pertanyaan | Jawaban singkat | Status |
|---|---|---|
| Mengapa REG_273 bukan titik awal R2? | Belum ada alasan tertulis. | Tim harus memutuskan jawabannya |
| Mengapa peringkat tambahan absolut stabil? | Pada COP 3 peringkatnya sama dengan peringkat volume gas, jadi sebagian karena konstruksi. | Jawaban ada |
| Apakah gas = EUI x luas ditegakkan? | Tidak di prediksi (rasio 0,78 sampai 1,08). | Jawaban ada |
| Notebook menulis "terbesar", slide tidak. | Lihat baris "Bila ditanya" di catatan slide 5. | Jawaban ada |
| Mengapa ada dua versi notebook? | Versi 23 Okt belum memuat subjudul 2.1 sampai 2.3; kode, output, dan In [n] sama. | Jawaban ada |
| Alasan memakai log target? | Belum tertulis. | Tim harus mengisi |

## P3. Rekayasa fitur dan pra-pemrosesan (25 poin)

Per slide cukup tiga hal: fitur apa yang dibuat, mengapa masuk akal secara fisik, dan bagaimana diperiksa.
- Fitur: rasio fisik (porsi 25 sampai 30%) dan fitur kNN spasial yang dihitung out-of-fold.
- Pemeriksaan: uji permutasi label lulus (0 baris berubah; kontrol bocor mengubah 42.931 baris).
- Urutan subzona: 96,5% pasangan subzona berurutan berada di zona yang sama.
- Jangan ucapkan: "terbesar" untuk fitur kNN. Rata-rata target wilayah memberi penurunan galat yang sama atau lebih
  besar.

## P4. Arsitektur model dan notebook (15 poin)

- Satu kalimat: "43 model = 41 blended + 2 LightGBM terpisah".
- Hafal peta notebook: 56 sel (37 kode, 19 markdown), 17 modul, rujukan di baris kaki berupa nomor bagian dan In [n].
- Isi dalam sel tidak perlu dikuasai.

## Dilewati dengan sengaja

- Desain CV, bobot blending, dan eksperimen pseudo-label (tidak disebut di deliverable, sesuai arahan).
- Angka COP per iklim selain tiga zona di slide 10.
- Klaim yang terbukti tidak benar dan tidak boleh diucapkan: "terbesar" untuk gain fitur kNN; "Q2 punya tekanan gardu
  terkecil" (Q4 +87,04% lebih rendah dari Q2 +90,24%, dan tidak ada data puncak atau kapasitas).

## Urutan kerja malam ini dan besok pagi

1. Putuskan pembicara per bagian dan isi `[PERLU DIISI]` di deck.
2. Baca P1 sampai paham rantainya, lalu ucapkan R1, R2, R3 tanpa melihat slide.
3. Latih satu putaran penuh dengan pengatur waktu 9:30.
4. Ucapkan enam jawaban tanya jawab di atas dengan suara keras.
5. Buka `BISMILLAH_Final_v5.pptx` di PowerPoint asli (belum pernah diperiksa di sana) dan cek tata letak.
6. Besok pagi: satu putaran latihan singkat, tanpa materi baru.
