# deck/ppt02: *Deck* Final v3 (*top-down*, Prinsip Piramida Minto)

Folder ini berisi rancangan ulang presentasi final DICE FEST 2026 (Zoom, 30 Oktober 2026) menurut arahan pengguna
tanggal 27 Oktober 2026 (`00_arahan_minto.md`): struktur *slide* demi *slide*, narasi pembicara, durasi, *slide*
cadangan, daftar verifikasi, dan berkas PPTX yang dibangun dari struktur tersebut. *Deck* v2 di `../ppt01` tetap utuh
sebagai arsip.

## Berkas

| Berkas | Isi |
|---|---|
| `BISMILLAH_Final_v3.pptx` | *deck* presentasi: 13 *slide* utama (9 menit 30 detik) dan 11 *slide* cadangan C1 sampai C11, catatan pembicara berisi rentang waktu, pembicara, narasi lengkap, dan sumber (cadangan: pertanyaan juri, pedoman jawaban, penjawab) |
| `BISMILLAH_Final_v3_pratinjau.pdf`, `pratinjau/` | pratinjau setiap *slide* (PNG 1600 × 900) dari perender sendiri; lihat bagian "Batas pratinjau" |
| `pdf_hp/01_struktur_deck_minto.pdf` | versi baca dokumen struktur: *slide* 4:3 dengan daftar isi bertaut; setiap bagian *slide* memuat pratinjau *slide* PPTX-nya |
| `01_struktur_deck_minto.md` | sumber isi: identifikasi (tesis, rekomendasi, bukti, angka wajib, keterbatasan), piramida dan peta waktu, 13 *slide* utama, 11 *slide* cadangan, konflik dan hal yang perlu dikonfirmasi, daftar verifikasi, perubahan dari v2, dari audit, dan saat PPTX dibangun (G3) |
| `00_arahan_minto.md` | salinan arahan pengguna |
| `fakta_ppt02.md` | lembar fakta untuk angka yang tidak tercetak persis di lembar fakta lama, termasuk angka grafik PPTX (bagian E) |
| `make_visuals_v3.py`, `img/` | peta kuadran dan plot dependensi SHAP (PNG), serta `img/data_grafik.json` untuk grafik batang asli PowerPoint; setiap angka diperiksa dengan *assertion* terhadap keluaran *notebook* dan berkas bukti |
| `isi_deck.py` | mengekspor judul, isi layar, narasi, waktu, sumber, dan bukti cadangan dari dokumen struktur ke `build/isi_v3.json` |
| `build_deck_v3.js` | pembangun PPTX (pptxgenjs 4.0.1) dengan tema, tiga *layout* (GELAP, ISI, CADANGAN) berisi *placeholder* bernama, dan enam *section* |
| `cek_pptx.py` | pemeriksa PPTX terhadap dokumen: judul, setiap kata dan angka "Isi Teks", narasi di catatan, angka di layar, *section*, ukuran huruf, batas *slide*, em-dash |
| `render_pptx.py` | perender pratinjau PPTX ke PNG dan PDF dengan Carlito (metrik sama dengan Calibri), sekaligus pelapor teks meluap, kata terpotong, tabel memanjang, dan objek di luar *slide* |
| `cek_struktur.py` | pemeriksa dokumen struktur: durasi, kata lisan per menit, frasa wajib, jumlah model, angka terhadap lembar fakta, angka baru di menit terakhir, istilah teknis, kata kausal, em-dash; `--tulis` memperbarui blok B3 dan F |
| `bukti/cek_zona_contoh.py`, `.json` | posisi zona contoh pada skenario COP seragam dan COP per iklim, zona Q2 yang tetap Q2, dan urutan nomor subzona |
| `build_pdf.py` | pembangun PDF baca (salinan `../ppt01/build_pdf_hp.py` dengan daftar sumber v3 dan pratinjau *slide*) |

## Membangun ulang dan memeriksa

Alat yang dibutuhkan di luar `.venv` proyek:

- Node 24 dengan `pptxgenjs@4.0.1 react react-dom react-icons sharp`, misalnya
  `npm install --prefix /tmp/pptx-node pptxgenjs@4.0.1 react react-dom react-icons sharp`.
- Skill pptx (berisi `scripts/apply_theme.js` dan `scripts/office/validate.py`), ditunjuk dengan `PPTX_SKILL_DIR`.
- Python dengan `python-pptx lxml pillow defusedxml`, misalnya venv di `/tmp/pptx-py`.

```bash
.venv/bin/python deck/ppt02/bukti/cek_zona_contoh.py           # sekitar 30 detik
.venv/bin/python deck/ppt02/make_visuals_v3.py                  # sekitar 2 menit; img/
.venv/bin/python deck/ppt02/cek_struktur.py --tulis
.venv/bin/python deck/ppt02/isi_deck.py                        # build/isi_v3.json (judul dua baris seimbang)
NODE_PATH=/tmp/pptx-node/node_modules PPTX_SKILL_DIR=<skill pptx> node deck/ppt02/build_deck_v3.js
/tmp/pptx-py/bin/python deck/ppt02/cek_pptx.py
/tmp/pptx-py/bin/python <skill pptx>/scripts/office/validate.py deck/ppt02/BISMILLAH_Final_v3.pptx
/tmp/pptx-py/bin/python skills/pitch-deck-kompetisi/scripts/check_deck.py deck/ppt02/BISMILLAH_Final_v3.pptx \
    --facts deck/fakta_notebook.txt --facts deck/fakta_turunan.md --facts deck/ppt01/fakta_tambahan.md \
    --facts reports/WRITEUP_DRAFT_vB.md --facts deck/ppt02/fakta_ppt02.md --minutes 9.5
/tmp/pptx-py/bin/python deck/ppt02/render_pptx.py deck/ppt02/BISMILLAH_Final_v3.pptx --out deck/ppt02/pratinjau \
    --pdf deck/ppt02/BISMILLAH_Final_v3_pratinjau.pdf --strict
pdfenv/bin/python deck/ppt02/build_pdf.py
```

Isi *deck* diubah di `01_struktur_deck_minto.md`, bukan di PPTX, lalu urutan di atas dijalankan ulang. Tata letak
diubah di `build_deck_v3.js`.

## Rancangan PPTX

- **Tema dan *layout*.** Warna *deck* v2 ditulis ke tema (biru tua `16324F` untuk struktur, jingga `D97B29` untuk gas
  dan angka kunci, hijau kebiruan `1F7A8C` untuk listrik dan validasi, ungu `5B4A86` untuk interpretasi), huruf
  Calibri. *Layout* GELAP dipakai *slide* 1 dan 13, ISI untuk *slide* utama, CADANGAN untuk C1 sampai C11. Judul dan
  baris penanda bagian berada di *placeholder*, sehingga muncul di tampilan *outline* PowerPoint.
- **Motif.** Ikon dalam lingkaran berwarna; sudut segitiga trilema (emisi jingga, biaya ungu, jaringan hijau
  kebiruan) muncul lagi sebagai titik warna pada kartu rekomendasi *slide* 2 dan 13; rantai TEMUAN, INTERPRETASI,
  REKOMENDASI, dan LANGKAH berbentuk sama di *slide* 9 sampai 11.
- **Grafik.** Grafik batang *slide* 3, 9, 10, dan 11 adalah grafik asli PowerPoint tanpa teks sumbu. Label kategori
  dan nilai ditulis sebagai kotak teks di posisi batang yang dihitung dari area plot tetap (*manual layout*), sehingga
  koma desimal tetap tampil di komputer berbahasa apa pun. Peta 330 zona dan plot dependensi SHAP berupa gambar karena
  memuat ratusan sampai 20.000 titik.

## Hasil pemeriksaan (28 Oktober 2026)

- **`cek_pptx.py`:** 0 galat. Judul, setiap kata dan angka "Isi Teks", narasi lengkap di catatan, angka di layar,
  enam *section*, huruf minimal 10 pt (89 label dan keterangan berukuran 10 sampai 11 pt), dan semua objek di dalam
  *slide*.
- **`validate.py` (skill pptx):** lulus.
- **`check_deck.py` (skill pitch-deck):** 0 galat untuk PPTX dan untuk dokumen (rencana 9 menit 30 detik). Peringatan
  kepadatan muncul di setiap *slide* karena label grafik dan diagram ikut dihitung sebagai kata di layar.
- **`cek_struktur.py`:** kesepuluh butir daftar verifikasi arahan terpenuhi.
- **`render_pptx.py --strict`:** tidak ada teks meluap, kata terpotong, tabel memanjang, atau objek di luar *slide*.
- **Tinjauan visual berkonteks bersih.** Pemeriksa yang tidak ikut menyusun *deck* menemukan 10 cacat wajib dan
  sejumlah perbaikan kecil pada pratinjau pertama. Semuanya sudah ditangani, antara lain:
  - label sumbu SHAP yang salah: *tick* 0,25 tercetak "0,2". Cacat yang sama ada di `../ppt01/img/s08_dep_size.png`,
    yang tidak diubah;
  - tumpang tindih di C9 dan sisipan *slide* 11 yang terlalu kecil;
  - judul yang terpotong di tengah kata dan angka yang terpisah dari satuannya.

## Batas pratinjau

LibreOffice tidak tersedia di server ini, sehingga pratinjau dibuat oleh `render_pptx.py`. Huruf Carlito memiliki
metrik yang sama dengan Calibri, sehingga pemotongan baris semestinya sama dengan PowerPoint, tetapi hasil akhirnya tetap
perlu dilihat di PowerPoint atau Google Slides sebelum tampil.

## Yang masih perlu dilakukan tim

1. Buka `BISMILLAH_Final_v3.pptx` di PowerPoint, periksa setiap *slide* dalam mode presentasi, lalu ekspor PDF
   cadangan.
2. Isi nama ketiga pembicara pada penanda "[nama Orang 1]" sampai "[nama Orang 3]" di catatan *slide* 2, 7, dan 11
   (di dokumen struktur juga, lalu bangun ulang).
3. Jawab butir [PERLU KONFIRMASI] di bagian E dokumen struktur: ketentuan pembicara, cara menghitung slot 10 menit,
   dan jam tampil.
4. Latih penuh dua kali dengan pengatur waktu memakai tampilan penyaji (catatan memuat rentang waktu setiap *slide*).
