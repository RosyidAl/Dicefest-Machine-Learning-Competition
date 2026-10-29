# deck/ppt03: *Deck* Final v4

Folder ini berisi *deck* v4 presentasi final DICE FEST 2026 (Zoom, 30 Oktober 2026). Versi ini merevisi *deck* v3
(`../ppt02`) atas permintaan pengguna tanggal 28 Oktober 2026, dengan beberapa elemen diambil dari *deck* v2
(`../ppt01`). Alur *top-down*, durasi 9 menit 30 detik, pembagian tiga pembicara, dan 11 *slide* cadangan tetap sama
dengan v3. Folder `../ppt01` dan `../ppt02` tidak diubah.

## Berkas

| Berkas | Isi |
|---|---|
| `BISMILLAH_Final_v4.pptx` | *deck* presentasi: 13 *slide* utama (9 menit 30 detik) dan 11 *slide* cadangan C1 sampai C11. Catatan pembicara berisi rentang waktu, peran pembicara, narasi lengkap, dan sumber. Catatan *slide* cadangan berisi pertanyaan juri dan pedoman jawaban |
| `BISMILLAH_Final_v4_pratinjau.pdf`, `pratinjau/` | pratinjau setiap *slide* (PNG 1600 × 900) dari perender sendiri; lihat bagian "Batas pratinjau" |
| `01_struktur_deck_v4.md` | sumber isi: identifikasi, piramida dan peta waktu (B4 memuat padanan indeks sel dan In [n]), 13 *slide* utama (dengan medan "Sel notebook" untuk baris kaki), 11 *slide* cadangan, konflik yang ditangani (E), daftar verifikasi (F), perubahan v4 per kriteria (G4), dan perbaikan setelah tinjauan independen (G5) |
| `fakta_ppt03.md` | lembar fakta untuk angka yang baru di v4 |
| `make_visuals_v4.py`, `img/` | peta kuadran, plot dependensi SHAP (termasuk dua gambar baru: listrik per rumah untuk *slide* 10 dan adopsi versi besar untuk *slide* 11), dan `img/data_grafik.json` untuk grafik asli PowerPoint. Setiap angka diperiksa dengan *assertion* terhadap keluaran *notebook* dan berkas bukti |
| `isi_deck.py` | mengekspor isi dokumen ke `build/isi_v4.json` |
| `build_deck_v4.js` | pembangun PPTX (pptxgenjs 4.0.1), turunan `../ppt02/build_deck_v3.js` |
| `cek_struktur.py` | pemeriksa dokumen: durasi, kata lisan per menit, angka terhadap lembar fakta, frasa wajib, pemetaan gardu, baris PELAKSANA, penanda nama; `--tulis` memperbarui blok B3 dan F |
| `cek_pptx.py` | pemeriksa PPTX terhadap dokumen: judul, isi, narasi, angka, baris kaki nomor sel, ukuran huruf minimal 14 pt, rata-rata kata per *slide*, dan penanda nama di catatan |
| `render_pptx.py` | perender pratinjau (salinan v3 yang kini juga menggambar grafik garis) sekaligus pelapor teks meluap |

Berkas dari folder lain yang dipakai: `../ppt01/fonts/` (Carlito), `../ppt01/bukti/` dan `../ppt02/bukti/` (berkas
bukti), `../ppt02/fakta_ppt02.md` (lembar fakta), serta *notebook* final dan data mentah untuk `make_visuals_v4.py`.

## Perubahan dari v3

Rincian lengkapnya ada di bagian G4 dokumen struktur.

- **Kriteria 2 (arsitektur):**
  - judul *slide* 6 diambil dari v2;
  - label "Jalur 3 · Adopsi" diperbaiki menjadi "Porsi bersih dan adopsi: tanpa model besar", karena jalur ini juga
    memuat porsi bersih;
  - kartu sidik sha256 di *slide* 7 diganti grafik garis *leaderboard* publik per langkah (sel 37) dengan titik skor
    privat; sidiknya pindah ke baris kaki;
  - setiap *slide* utama mencetak rujukan *notebook* di baris kaki, ditulis seperti yang terlihat juri: nomor bagian
    untuk sel markdown dan nomor eksekusi In [n] untuk sel kode (contoh "Notebook: bagian 7.2, In [35]"). Dokumen
    dan catatan pembicara tetap memakai indeks sel dari 0; padanannya ada di bagian B4.
- **Kriteria 1 (pra-pemrosesan):**
  - pita bawah *slide* 6 kini berisi tiga fakta struktur data dari v2, lengkap dengan angka buktinya dari sel 25;
  - *slide* 3 menampilkan porsi kontribusi SHAP rasio fisik turunan 25–30%, dihitung ulang dari nilai SHAP dan sama
    dengan *writeup* Tabel 3;
  - batang fitur turunan di grafik *slide* 3 diberi warna sendiri dengan legenda.
- **Kriteria 4 (kerapihan):**
  - rata-rata kata di layar turun dari 118 menjadi 97,3 per *slide* (utama 95,2, cadangan 99,8). Baris "Menjawab"
    pindah ke catatan, dan tabel "kalimat aman" C5 pindah ke pedoman jawaban;
  - semua teks di layar minimal 14 pt; baris kaki 12 pt;
  - *slide* 13: langkah berikutnya kini sejajar dengan kartu rekomendasinya. Di v3, langkah gardu berada di bawah
    kartu *retrofit*;
  - *slide* 23 (C10): rujukan "contoh *deck* v2" dihapus, kuadran REG_263 diisi, tabel disusun per ukuran dengan
    format "COP 3 → COP per iklim", dan titik awal ditebalkan;
  - catatan pembicara: penanda "[nama Orang N]" diganti kalimat serah terima tanpa nama, dan jalur berkas bukti
    ditulis lengkap.
- **Kriteria 3 (kebijakan):**
  - *slide* 10 mendapat dasar SHAP: panel listrik per rumah dari v2 yang digambar ulang (sumbu v2 mencetak 0,25
    sebagai "0,2");
  - grafik SHAP adopsi menjadi visual utama *slide* 11;
  - baris "Pelaksana:" ada di *slide* 9 sampai 11;
  - pemetaan gardu disamakan: R2 berada di Q1 dan Q3 pada *slide* 2, 8, 10, 13, dan C6, sesuai sel 51 butir 3.

## Membangun ulang dan memeriksa

Alat di luar `.venv` proyek sama dengan v3:
- Node 24 dengan `pptxgenjs@4.0.1 react react-dom react-icons sharp`;
- skill pptx (`PPTX_SKILL_DIR`);
- Python dengan `python-pptx lxml pillow defusedxml`.

```bash
.venv/bin/python deck/ppt03/make_visuals_v4.py                  # sekitar 2 menit; img/
.venv/bin/python deck/ppt03/cek_struktur.py --tulis
.venv/bin/python deck/ppt03/isi_deck.py                         # build/isi_v4.json
NODE_PATH=<node_modules> PPTX_SKILL_DIR=<skill pptx> node deck/ppt03/build_deck_v4.js
<python-pptx> deck/ppt03/cek_pptx.py
<python-pptx> <skill pptx>/scripts/office/validate.py deck/ppt03/BISMILLAH_Final_v4.pptx
<python-pptx> ~/.claude/skills/pitch-deck-kompetisi/scripts/check_deck.py deck/ppt03/BISMILLAH_Final_v4.pptx \
    --facts deck/fakta_notebook.txt --facts deck/fakta_turunan.md --facts deck/ppt01/fakta_tambahan.md \
    --facts reports/WRITEUP_DRAFT_vB.md --facts deck/ppt02/fakta_ppt02.md --minutes 9.5
<python-pptx> deck/ppt03/render_pptx.py deck/ppt03/BISMILLAH_Final_v4.pptx --out deck/ppt03/pratinjau \
    --pdf deck/ppt03/BISMILLAH_Final_v4_pratinjau.pdf --strict
```

Isi *deck* diubah di `01_struktur_deck_v4.md`, tata letak di `build_deck_v4.js`.

## Hasil pemeriksaan (28 Oktober 2026)

- **`cek_struktur.py`:**
  - kesepuluh butir verifikasi terpenuhi;
  - kecepatan narasi 125 sampai 139 kata lisan per menit;
  - 0 angka di luar lembar fakta;
  - pemetaan gardu sama di *slide* 2, 8, 10, dan 13;
  - baris PELAKSANA ada di *slide* 9 sampai 11;
  - tidak ada penanda nama.
- **`cek_pptx.py`:**
  - 0 galat;
  - rata-rata 97,3 kata di layar per *slide*;
  - tidak ada teks di bawah 14 pt selain baris kaki 12 pt;
  - baris kaki setiap *slide* utama sama dengan medan "Sel notebook".
- **`validate.py` (skill pptx):** lulus.
- **`check_deck.py` (skill pitch-deck):**
  - 0 galat;
  - peringatan kepadatan, karena pedoman skill 45 kata per *slide*; batas yang diminta pengguna 100 kata;
  - peringatan tabel cadangan lebih dari 5 baris.
- **`render_pptx.py --strict`:** tidak ada teks meluap, kata terpotong, atau tabel memanjang.
- **Tinjauan independen.** Pemeriksa berkonteks bersih memeriksa semua angka, catatan, dan pratinjau. Kesimpulannya:
  kesepuluh permintaan sudah dikerjakan dan semua angka cocok dengan lembar fakta. Ke-18 temuannya diperiksa ulang ke
  sumber, lalu temuan yang terbukti diperbaiki (bagian G5). Yang terpenting, rujukan sel di baris kaki kini memakai
  penanda yang terlihat di *notebook*.

## Batas pratinjau

LibreOffice tidak tersedia di server ini, sehingga pratinjau dibuat oleh `render_pptx.py`. Huruf Carlito memiliki
metrik yang sama dengan Calibri, jadi pemotongan barisnya semestinya sama dengan PowerPoint. Ada dua hal yang perlu
dilihat langsung di PowerPoint:
- grafik garis asli di *slide* 7, yang baru di v4;
- pangkat "−16" di *slide* 6, yang memakai format superskrip.

## Yang masih perlu dilakukan tim

1. Buka `BISMILLAH_Final_v4.pptx` di PowerPoint, periksa setiap *slide* dalam mode presentasi, lalu ekspor PDF
   cadangan.
2. Bila ingin menyebut nama saat serah terima, tambahkan nama di akhir narasi *slide* 2, 7, dan 11. Catatan sekarang
   memakai kalimat tanpa nama.
3. Latih penuh dua kali dengan pengatur waktu memakai tampilan penyaji.
