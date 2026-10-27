# Struktur Deck Final v2 (10 menit): Tim BISMILLAH, DICE FEST 2026

Dokumen ini adalah sumber kebenaran isi deck final. Bagian C dan D dibangkitkan otomatis oleh `build_deck_v2.py` dari
data yang sama dengan `BISMILLAH_Final_v2.pptx`, sehingga isi dokumen dan isi slide tidak bisa berbeda. Bagian A, B,
E, dan F ditulis tangan (`brief_kepala.md`, `brief_ekor.md`). Berkas ini menggantikan `../04_struktur_deck.md`.

**Sumber angka:** keluaran notebook `final/deliverables/DICEFEST2026_BISMILLAH.ipynb` (`../fakta_notebook.txt`),
`../fakta_turunan.md`, dan `fakta_tambahan.md`. Nomor sel dihitung dari 0. Setiap slide menyebut nomor selnya di label
bukti.

## A. Perubahan dari deck 04 dan alasannya

| Celah di deck 04 | Perubahan di v2 | Slide |
|---|---|---|
| 1. Naskah 01 dan 02 terlalu panjang dan bersaing dengan narasi 04 | Narasi tiap slide kini satu-satunya naskah lisan; naskah 01 dan 02 ditandai arsip di `README.md` | semua |
| 2a. Pembuka langsung memakai skor dan jargon | Slide 1 berisi kesimpulan (urutan program untuk 330 zona). Slide 2 menjelaskan mengapa masalah layak diselesaikan, dengan risiko kenaikan listrik +94,85% | 1, 2 |
| 2b. Slide arsitektur memuat lima pesan dalam 80 detik | Slide 6 kini satu pesan: tiga jalur untuk lima target, dengan galat jujur per jalur. Rincian bobot adaptif, koreksi, dan penyesuaian akhir pindah ke cadangan C1 sampai C3 | 6, C1, C2, C3 |
| 3. Rekomendasi belum menyebut pelaksana, ukuran keberhasilan, dan tingkat keyakinan | Tiap rekomendasi punya slide sendiri dengan kartu Siapa, Apa, Ukuran, dan Dasar bukti. Tabel peta menghubungkan tiap kuadran ke rekomendasinya. Uji ketahanan menunjukkan urutan zona tahan galat prediksi tetapi peka terhadap COP per iklim, sehingga daftar zona disebut kandidat | 9 sampai 12, C4 |
| 4. Keterbatasan dan penutup menumpang di slide rekomendasi 3 | Slide 13 khusus keterbatasan (tiga hal yang belum diketahui beserta penanganannya), slide 14 khusus penutup (tiga alasan dan tiga langkah berikutnya) | 13, 14 |
| 5. Klaim bebas kebocoran hanya didukung selisih LB 0,00011 | Uji label acak pada modul notebook, untuk baris validasi (42.931) dan baris latih (77.275): 0 baris fitur berubah; kontrol bocor berubah di 42.931 baris | 5, C7 |
| 6. LB publik dipakai sebagai bukti utama | Slide 7 memakai skor privat 0,24844 (peringkat 8) dari data yang tidak pernah terlihat, ditambah AUC adversarial 0,4997 dan 8,14% vs 8,17%; *leaderboard* publik hanya disebut sebagai konfirmasi arah | 7 |
| 7. Lima bagian RISIKO di tanya jawab belum ditutup | P4 ditutup dengan uji label acak; P2 dihitung ulang; P1 didukung bukti pengembangan (D002); P3 dan P5 diberi rujukan audit. Lihat `06_qna_juri_v2.md` | C1, C7 |
| 8. Belum ada deck yang bisa dipresentasikan | `BISMILLAH_Final_v2.pptx` (dapat disunting, dengan catatan pembicara) dan pratinjau PDF | semua |

**Sistem bukti di layar.** Setiap angka penting diberi label kecil yang sekaligus menyebut sumbernya:
- **FAKTA** (biru tua): keluaran notebook atau data;
- **HASIL** (hijau kebiruan): hasil validasi atau model penjelas;
- **ASUMSI** (jingga): asumsi skenario;
- **INTERPRETASI** (ungu): bacaan atau usulan tim.

Label ini menjawab arahan A6 dan C5 (membedakan fakta, asumsi, dan interpretasi) tanpa menambah paragraf.

**Koreksi setelah audit independen (27 Oktober).** Pemeriksa berkonteks bersih menemukan 25 masalah. Yang terpenting
sudah diperbaiki, dan angka barunya ada di `fakta_tambahan.md` bagian F sampai H:
- **Potensi *retrofit*:** 7,39 TWh kini disebut batas atas. Definisinya (sel 48) ikut menghitung ukuran rumah; bila
  hanya periode bangunan dihitung, tersisa sekitar separuh.
- **Rekomendasi 2:** dasarnya kini rumus kenaikan (gas dibanding listrik saat ini) dan tambahan absolut (REG_263),
  bukan "rumah besar".
- **Rekomendasi 3:** kini pilot pompa panas di zona Q2 (REG_249), dengan tingkat adopsi 40% vs 2,73%. Klaim keliru
  "adopsi hampir hanya di kawasan tanpa gas" sudah dihapus.
- **Urutan zona:** klaim "tetap" diganti hasil uji ketahanan. Urutan tahan galat prediksi (Spearman 0,998) tetapi peka
  terhadap COP per iklim (0,81).
- **Slide 7:** *leaderboard* publik hanya sebagai konfirmasi arah (penolakan D011 disebut); privat "tidak memburuk".
- **Slide 8:** ukuran rumah ±37 vs ±173 m² untuk pita 5% ekstrem; kontras kuartil +34% dan +35%.
- **Lain-lain:**
  - judul slide 1 dan 2;
  - label galat jujur "sebelum koreksi" di slide 6;
  - baris notebook di slide 6;
  - "perangkat yang sama" untuk klaim identik byte;
  - bobot 224 + 224 + 104 di C1;
  - asal S1, S3, dan S4 di C3;
  - urutan angka D002 di C7;
  - kata "privat".

**Desain.**
- **Rasio dan huruf:** 16:9, Calibri. Pratinjau memakai Carlito, yang ukuran hurufnya identik.
- **Ukuran teks:** judul 28 pt, isi 14 sampai 18 pt, label dan istilah 11 sampai 12 pt.
- **Warna:**
  - biru tua `#16324F` untuk struktur;
  - hijau kebiruan `#1F7A8C` untuk listrik dan hasil validasi;
  - jingga `#D97B29` hanya untuk angka terpenting dan gas.
- **Slide gelap:** hanya pembuka dan penutup.
- **Dekorasi:** tidak ada garis hias atau ikon dekoratif.

## B. Pembagian waktu dan rubrik

Kecepatan acuan 130 kata per menit. Total rencana 9 menit 20 detik, menyisakan 40 detik dari batas 10 menit untuk
serah terima antarpembicara dan jeda.

| Bagian rubrik (bobot) | Slide | Waktu | Porsi dari isi (tanpa pembuka dan penutup) |
|---|---|---|---|
| Kebijakan dari interpretabilitas (35) | 8, 9, 10, 11, 12 | 3 menit 40 detik | 47% |
| Rekayasa fitur dan pra-pemrosesan (25) | 3, 4, 5, separuh 7 | 2 menit 17 detik | 30% |
| Arsitektur model (15) | 6, separuh 7 | 1 menit 12 detik | 16% |
| Keterbatasan (bagian dari kebijakan) | 13 | 35 detik | 7% |
| Pembuka dan penutup | 1, 2, 14 | 1 menit 35 detik | |

Slide 7 dihitung separuh untuk fitur, karena membuktikan validitas pra-pemrosesan, dan separuh untuk arsitektur, karena
memperlihatkan sumbangan tiap komponen. Bobot slide dan tanya jawab (25) dinilai dari keseluruhan deck dan sesi tanya
jawab. Kesiapan tanya jawab ada di `06_qna_juri_v2.md` dan cadangan C1 sampai C8.
