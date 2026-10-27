# deck/ppt01: Deck Final v2 (babak final DICE FEST 2026, 30 Oktober 2026)

Folder ini adalah sumber kebenaran untuk presentasi final. Semua berkas di luar folder ini yang membahas deck dianggap
arsip, kecuali yang ditandai "rujukan" di tabel kedua.

## Berkas aktif

| Berkas | Isi |
|---|---|
| `BISMILLAH_Final_v2.pptx` | deck 16:9: 14 slide utama dan 8 slide cadangan (C1 sampai C8). Huruf Calibri, bisa disunting. Catatan pembicara berisi narasi dan sumber |
| `BISMILLAH_Final_v2_pratinjau.pdf` | pratinjau raster (1920 × 1080) dari perender `deckkit.py`; hanya untuk memeriksa tata letak, bukan berkas presentasi |
| `05_struktur_deck_v2.md` | struktur dan narasi per slide, perubahan dari deck 04, pembagian waktu sesuai rubrik, dan daftar periksa arahan. Menggantikan `../04_struktur_deck.md` |
| `06_qna_juri_v2.md` | status lima bagian RISIKO dari `../03_qna_juri.md` beserta bukti baru, bank 18 pertanyaan dengan jawaban singkat, dan pembagian penjawab untuk tiga orang |
| `07_kartu_latihan.md` | kartu latihan tiga orang: peran, urutan dan kalimat serah terima, naskah lengkap per orang, alur tanya jawab, daftar periksa Zoom, alternatif dua pembicara |
| `pdf_hp/` | bahan belajar berbentuk slide PDF 4:3 (10 × 7,5 inci, teks 18 pt) dari 05, 07, 06, `../03_qna_juri.md`, `../02_naskah_kebijakan.md`, dan `fakta_tambahan.md`. Setiap PDF punya sampul, daftar isi yang bisa diketuk, dan satu bagian per slide. Bagian naskah di 05 dan 07 memuat pratinjau slide deck yang dibahas |
| `fakta_tambahan.md` | lembar fakta untuk angka di luar keluaran notebook (skor privat, hasil uji baru, angka turunan) |
| `bukti/uji_kebocoran_notebook.py`, `.json`, `.log` | uji label acak untuk baris validasi dan uji pada modul notebook: 0 baris fitur berubah, kontrol bocor berubah di 42.931 baris (lulus) |
| `bukti/uji_kebocoran_oof_dalam.py`, `.json`, `.log` | uji label acak untuk baris latih: label satu bagian dalam (77.275 baris) diacak, fitur baris itu identik bit, 309.097 baris lain berubah (lulus) |
| `bukti/skor_jujur_bobot_adaptif.py`, `.json`, `.log` | skor jujur bobot adaptif A2xM2xp2 dengan fungsi notebook; skor global mereproduksi sel 35 persis |
| `bukti/cek_kebijakan.py`, `.json`, `.log` | pemeriksaan setelah audit: adopsi per porsi gas jaringan, uji ketahanan urutan zona (galat prediksi dan COP per iklim), komposisi potensi *retrofit*, kontras ukuran rumah, tambahan listrik absolut |
| `make_visuals_v2.py`, `img/` | grafik slide. Setiap grafik diperiksa dengan *assertion* terhadap keluaran notebook (sel 45, 49, 50) sebelum disimpan |
| `build_pdf_hp.py` | pembangun `pdf_hp/` (pdfenv: markdown-it-py, reportlab, dan Pillow). Memakai `pratinjau/`, jadi jalankan setelah `build_deck_v2.py`. Hasilnya identik byte setiap dibangun ulang |
| `build_deck_v2.py`, `deckkit.py`, `brief_kepala.md`, `brief_ekor.md`, `fonts/` | generator deck: satu sumber isi untuk PPTX, pratinjau, `05`, dan `07`. Folder `fonts/` berisi Carlito, huruf terbuka yang ukurannya identik dengan Calibri (lisensi SIL OFL 1.1, `fonts/OFL.txt`) |

## Status berkas lama di `deck/`

| Berkas | Status | Keterangan |
|---|---|---|
| `01_naskah_fitur_arsitektur.md` | arsip | narasinya sudah dipangkas ke slide 3 sampai 7 di `05` |
| `02_naskah_kebijakan.md` | arsip, tetapi bagian A sampai C tetap jadi rujukan | narasinya sudah dipangkas ke slide 8 sampai 13 di `05`; tabel arti `hvac_*` dan bacaan grafik tetap berguna saat tanya jawab |
| `03_qna_juri.md` | rujukan | pedoman jawaban rinci P1 sampai P5; status RISIKO diperbarui di `06` |
| `04_struktur_deck.md`, `04_struktur_deck.pdf`, `build_04_pdf.py` | arsip | digantikan `05` |
| `BISMILLAH_Pitch_Deck_v1.pdf`, `build_deck.py`, `notes.md` | arsip, jangan dipakai | versi 24 Oktober dengan struktur dan angka lama |
| `img/`, `make_visuals.py` | arsip | v2 memakai `ppt01/img/` |
| `fakta_notebook.txt`, `fakta_turunan.md` | rujukan | lembar fakta utama untuk `check_deck.py` |
| `jawaban.md` | rujukan | jawaban persiapan final |

Berkas lama tidak dipindahkan supaya tautan dan riwayat git tetap utuh.

## Membangun ulang

```bash
# 1. grafik (venv proyek: pandas, matplotlib, modul notebook)
.venv/bin/python deck/ppt01/make_visuals_v2.py
# 2. deck, pratinjau, 05 dan 07 (venv apa pun yang memiliki python-pptx dan Pillow)
python deck/ppt01/build_deck_v2.py
# 3. pemeriksaan
python ~/.claude/skills/pitch-deck-kompetisi/scripts/check_deck.py deck/ppt01/05_struktur_deck_v2.md \
    --facts deck/fakta_notebook.txt --facts deck/fakta_turunan.md --facts deck/ppt01/fakta_tambahan.md --minutes 10
python ~/.claude/skills/pitch-deck-kompetisi/scripts/check_deck.py deck/ppt01/BISMILLAH_Final_v2.pptx \
    --facts deck/fakta_notebook.txt --facts deck/fakta_turunan.md --facts deck/ppt01/fakta_tambahan.md
# 4. slide PDF 4:3 bahan belajar (setelah langkah 2, karena memakai pratinjau/)
pdfenv/bin/python deck/ppt01/build_pdf_hp.py
```

Uji bukti dijalankan dari akar repo dengan `.venv/bin/python deck/ppt01/bukti/<skrip>.py`. Durasinya:
- uji kebocoran validasi sekitar 20 menit;
- uji kebocoran baris latih sekitar 11 menit;
- skor jujur sekitar 4 menit;
- cek kebijakan sekitar 2 menit.

RAM yang dipakai di bawah 6 GB.

## Hasil pemeriksaan (27 Oktober 2026)

- **Audit independen:** pemeriksa berkonteks bersih menemukan 25 masalah, 5 berat, terutama di bagian kebijakan. Semua
  temuan berat dan menengah sudah diperbaiki (`05` bagian A).
- **`check_deck.py` pada `05_struktur_deck_v2.md`:**
  - 0 galat;
  - total rencana 9 menit 20 detik untuk 14 slide.
  - Peringatan kepadatan muncul di 14 slide. Teks prosa per slide 56 sampai 99 kata, lebih dari pedoman 45 kata. Teks itu
    tersusun dalam kartu dan label pendek, bukan paragraf. Slide terpadat adalah slide 5, karena diagram *cross-fit*
    ikut dihitung.
- **`check_deck.py` pada PPTX:**
  - 0 galat angka: setiap angka di 22 slide dan catatan pembicara ada di lembar fakta;
  - tidak ada em-dash.
- **`validate.py` (skill pptx):** semua validasi lulus. Yang diperiksa: skema, relasi, tipe konten, dan slide.
- **Ukuran teks:** 0 kotak teks yang meluap, dihitung dengan metrik Carlito yang sama dengan Calibri.
- **Pratinjau visual:** ke-22 slide sudah diperiksa satu per satu.

## Yang belum terverifikasi dan perlu dilakukan tim

1. **Render dengan PowerPoint.** Pratinjau dibuat oleh perender sendiri karena konversi dengan LibreOffice tidak
   diizinkan di server ini. Buka PPTX di PowerPoint, lalu periksa sekilas setiap slide, terutama:
   - pangkat "−16" di slide 3, yang memakai format superskrip PowerPoint;
   - simbol "≈" dan "±".

   Setelah itu ekspor PDF dari PowerPoint (File, Export, PDF) sebagai cadangan presentasi Zoom.
2. **Latihan dua kali penuh dengan pengatur waktu,** memakai `pdf_hp/07_kartu_latihan.pdf`. Target 9 menit 20 detik;
   batas keras 10 menit.
3. **Pembagian peran tiga orang** (07 bagian A):
   - Orang 1 (jangkar): slide 1, 2, dan 14, operator slide, dan pengarah tanya jawab;
   - Orang 2 (teknis): slide 3 sampai 7;
   - Orang 3 (kebijakan): slide 8 sampai 13.

   Bila panduan lomba mewajibkan semua anggota berbicara, pembagian ini sudah memenuhinya.
