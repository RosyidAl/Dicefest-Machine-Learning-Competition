# Laporan revisi *deck* v4 ke v5 (29 Oktober 2026)

## Ringkasan

| Berkas | Isi |
|---|---|
| `deck/ppt04/BISMILLAH_Final_v5.pptx` | 25 *slide*: 13 utama, cadangan C1 sampai C11, dan C12 baru di akhir |
| `deck/ppt04/edit_v5.py` | satu skrip python-pptx yang menyalin v4 lalu menerapkan tugas A sampai H dan satu putaran perbaikan hasil tinjauan. Bila dijalankan ulang, skrip menghasilkan berkas yang sama dari v4 |
| `deck/ppt04/laporan_v5.md` | laporan ini |

Menjalankan ulang: `<python dengan python-pptx 1.0.2, lxml, xlsxwriter> deck/ppt04/edit_v5.py`. Skrip memeriksa sidik
sha256 v4 sebelum dan sesudah penyuntingan (`e9177bf6…c057`, tidak berubah).

## Sumber yang dibaca

- **Notebook.** `final/DICEFEST2026_BISMILLAH.ipynb` tidak menyimpan output, dan seluruh *execution_count*-nya kosong.
  - Output dan nomor In [n] dibaca dari `final/deliverables/DICEFEST2026_BISMILLAH.ipynb`, yang sumber selnya identik
    sel per sel (56 sel: 37 kode, 19 markdown) dengan output dan In [1] sampai In [37].
  - Karena *execution_count* tersimpan di salinan itu, baris kaki memakai In [n], bukan "sel ke-k".
  - Padanannya: In [n] adalah sel kode ke-n; sel kode yang dirujuk adalah sel ke-(indeks + 1), misalnya In [35] = sel
    ke-51.
- **Versi notebook yang dikirim ke Kaggle.** Versi 23 Oktober (commit `27c8f2a`, sha256 `864f92cd…`) identik dalam
  kode, output, dan In [n]. Perbedaannya ada di sel markdown 28 dan 30, yang diubah 27 Oktober: subjudul 2.1 sampai
  2.3 dan blok "Cara membaca diagram" belum ada. Karena itu rujukan bagian di baris kaki v5 ditulis "bagian 2", yang
  benar untuk kedua versi.
- **Skrip cek terpisah** yang sudah ada di repositori:
  - `deck/ppt01/bukti/cek_kebijakan.py` (dengan `.json`);
  - `deck/ppt02/bukti/cek_zona_contoh.py` (dengan `.json`). Tambahan listrik absolut pada skenario COP per iklim tidak
    tersimpan di JSON mana pun. Angka itu diambil dengan menjalankan salinan skrip ini di folder sementara. Penulisan
    JSON dibuang dari salinan, sehingga tidak ada berkas repositori yang ditulis. Semua *assertion* lulus dan hasil
    JSON-nya identik dengan berkas di repositori.
- **Deck v4** hanya dibaca. Tidak ada notebook yang dijalankan.

## Tabel fakta

| Angka (dipakai di) | Sumber | Status |
|---|---|---|
| 757, 442, 434 GWh: tambahan listrik tahunan REG_263, REG_045, REG_270 pada COP 3 (slide 10) | `cek_kebijakan.json` kunci `tambahan_absolut` (0,757; 0,442; 0,434 TWh), dibuat `cek_kebijakan.py` baris 126-131; sama dengan gas zona In [35] × 0,8 / 3 (asumsi bagian 7.2) | cocok |
| 805, 525, 514 GWh: tambahan listrik pada COP per iklim (slide 10) | `cek_zona_contoh.py` baris 55-56 (COP 3,5 sampai 2,5 menurut peringkat iklim zona) dan 64-65 (`Z.gas * 0.8 / cop`); nilai tepat 805,45; 524,74; 513,77 GWh dari salinan baca-saja | turunan skrip cek |
| COP 2,5–3,5 per iklim (legenda slide 10) | `cek_zona_contoh.py` baris 56; tercetak di C10 | cocok |
| peringkat tambahan absolut REG_263 1→1, REG_045 3→3, REG_270 4→4 (slide 10) | `cek_zona_contoh.json` `zona_contoh.<zona>.peringkat_tambahan_absolut_dasar/_cop_iklim` (baris 64-65, 77-78) | cocok dengan prompt |
| REG_290 1→47, peringkat kenaikan persen (slide 10) | `cek_zona_contoh.json` `peringkat_kenaikan_dasar/_cop_iklim` (baris 62-63); peringkat 1 juga tampak di In [35] | cocok dengan prompt |
| zona peringkat 2 tambahan absolut: REG_273 (616 → 707 GWh) | `cek_kebijakan.json` (0,616 TWh), salinan baca-saja `cek_zona_contoh.py` | tidak dipakai di layar (lihat pertanyaan juri) |
| Q2: intensitas gas di atas median dan kenaikan listrik di bawah median (slide 11) | `explain.py` In [17] (kuadran dibelah median kedua sumbu); median kenaikan 94,85% di In [35]; zona Q2 tertinggi 94,59% | terbukti untuk setiap zona |
| rata-rata kenaikan per kuadran Q1 +102,49%, Q2 +90,24%, Q3 +99,10%, Q4 +87,04% | In [34] | membantah "paling kecil" di prompt |
| 9,46% (gas jaringan 60–80%) vs 2,73% (>80%, 70% kawasan) (slide 11) | `cek_kebijakan.json` `adopsi_per_pita_gas_jaringan` (baris 44-52); 70% = 70,08% kawasan latih | cocok |
| 40,24% dan 0,55% (opsi 2 prompt, tidak dipakai) | `cek_kebijakan.json` pita ≤40 | cocok, tidak dipakai |
| 56 sel, 17 modul, sakelar `DICEFEST_TRAIN`, ±5 jam (slide 6) | 56 sel = 37 kode + 19 markdown; 17 = sel `%%writefile` berkas .py (In [2] sampai In [18]), ditambah 2 berkas konfigurasi; In [1] baris 7 `TRAIN = os.environ.get("DICEFEST_TRAIN", "1") == "1"`; sel judul "Cara menjalankan": sekitar 5 jam bila prediksi model sudah ada di folder `WORK` | cocok dengan prompt |
| isi langkah 0,25025 → 0,24954 (slide 7) | catatan C2; `reports/cv_lb_history.csv` baris v12r sampai v15ncrkd: model adopsi, MLP dua keluaran dan salinan duplikat, pencabutan CatBoost, koreksi kNN (sekitar 0,0004) | cocok |
| ±37 dan ±173 m² (slide 3) | tidak ada di notebook; `cek_kebijakan.py` baris 163-173 (`kontras_ukuran_gas_per_m2`) | skrip cek terpisah |
| 96,5% (slide 4) | tidak ada di output notebook (In [36] mencetak 99,07% per baris); `cek_zona_contoh.py` baris 97-104; juga tercetak di PDF *writeup* bagian 4.3 | skrip cek terpisah |
| Spearman 0,998 dan 0,81 (slide 8) | tidak ada di notebook; `cek_kebijakan.py` baris 84-98 (`cop_per_iklim`, 0,8147) dan 100-124 (`derau_prediksi`, 0,9981) | skrip cek terpisah |
| +65% (slide 9) | notebook bagian 7.1 butir 1 ("sekitar 65%"), grafik In [33]; skrip memberi 66,5% untuk pita yang sama (notebook menamainya sekitar 30 dan 300 m²) | notebook |
| +34% antarkuartil (slide 9) | `cek_kebijakan.py` baris 170-176 (33,7%) | skrip cek terpisah |
| peringkat COP per iklim (slide 10) | `cek_zona_contoh.py` baris 55-65 | skrip cek terpisah |
| tingkat adopsi per kelompok dan REG_033 83,21%, 4,99% (slide 11) | `cek_kebijakan.py` baris 44-52; `cek_zona_contoh.json` `q2_tetap_q2` | skrip cek terpisah |
| −0,0022 (slide 5, kini di catatan) | notebook bagian 2 (sel ke-29): "menurunkan galat EUI model tunggal sebesar 0,0022" | cocok dengan teks notebook; nilai tepat eksperimen E0190 → E0191 adalah −0,00215 (pembulatan ganda di notebook) |
| "terbesar dari satu kelompok fitur" | notebook bagian 2 menyatakannya; ablasi pengembangan: rata-rata target wilayah memberi −0,0022 sampai −0,0034, sedangkan rasio fisik tidak pernah diablasi | tidak terbukti, kata tidak dipakai |
| konsistensi gas = EUI × luas lantai pada prediksi | `pipeline._assemble` In [13]: bobot gabungan, koreksi, dan penyesuaian akhir gas dan EUI terpisah; rasio gas/(EUI × luas) di submisi 0,78 sampai 1,08 | tidak ditegakkan (C.3 dilewati) |
| C12: NaN dibaca langsung XGBoost/LightGBM; MLP/NCA mengisi 0 plus penanda; fitur MLP/NCA dipotong ±6 simpangan baku; kawasan 2–4 meter dinaikkan; gas dekat 999.765 kWh ditarik turun; 10 dari 41 anggota menebak log | In [6] (`train_lib`), In [7] (`nn.preprocess`), bagian 3, In [13] (S1, S3), In [25] (tabel anggota), In [26] (41 model) | cocok |
| alasan target log | notebook hanya menulis bahwa grafik In [23] memakai skala log karena rentang nilainya lebar | tidak ada bukti alasan pemodelan → [PERLU DIISI] |
| 0,24913 setelah koreksi dan penyesuaian akhir (C2) | In [28] (tabel skor per langkah) | cocok; galat per target setelah koreksi tidak dicetak notebook |

## Perubahan per slide

Nomor *slide* mengikuti v4; isi setiap tugas dicocokkan ke *slide* yang memuatnya.

- **Slide 1 (G).** Kotak baru bergaya teks asumsi (15 pt) di bawah kolom kanan: "Pembicara: [PERLU DIISI: nama dan peran
  pembicara]". Tingginya 0,55 in supaya dua baris nama masih muat di atas baris kaki.
- **Slide 3, 4, 8, 9, 10, 11 (D).** Kotak baris kaki yang sudah ada diganti dengan bentuk "Sumber: notebook bagian …,
  In [n]; <angka>: skrip cek terpisah". Ukurannya tetap 12 pt seperti baris kaki lain. Kotaknya hanya melebar ke kiri,
  sehingga teks rata kanan tetap di posisi semula.
- **Slide 5 (E).**
  - "−0,0022" ternyata ada di *slide* 5, bukan *slide* 3 atau 4. Pita KONTRIBUSI kini berbunyi "perbaikan galat EUI
    model tunggal dari kelompok fitur ini; prediksi untuk 30% nilai peta zona memakainya", tanpa kata "terbesar".
  - Angkanya pindah ke catatan pada baris "Bila ditanya", yang tidak diucapkan.
  - Baris kakinya, "bagian 1, 2.2", merujuk subjudul yang tidak ada di notebook terkirim, jadi diganti "Sumber: notebook
    bagian 1 dan 2; uji label acak: skrip cek terpisah".
- **Slide 6 (C.1).**
  - Baris baru (14 pt, gaya kepala kolom galat): "56 sel notebook, 17 modul, satu sakelar pelatihan (DICEFEST_TRAIN);
    jalan ulang dari prediksi tersimpan ±5 jam". Kotaknya baru karena tidak ada kotak yang cocok.
  - Frasa "tanpa pelatihan" diganti "dari prediksi tersimpan", karena mode 5 jam membutuhkan prediksi 41 anggota yang
    sudah ada di `WORK`, dan prediksi itu tidak ikut dikirim.
  - Fakta pertama di pita bawah diberi "di data latih", karena identitas gas = EUI × luas lantai tidak ditegakkan pada
    prediksi (temuan C.3).
- **Slide 7 (C.2).** Label langkah kedua kini empat baris: "koreksi galat, / adopsi, MLP, / salin duplikat, / tanpa
  CatBoost". Kategori yang sama di data grafik dan *workbook* tertanam diganti dengan `chart.replace_data`, dan kotak
  label dilebarkan dari 1,121 menjadi 1,25 in dengan titik tengah yang sama.
- **Slide 10 (A).**
  - Grafik asli PowerPoint diganti datanya dengan `chart.replace_data`, sehingga jenis, warna dasar, celah batang,
    dan area plotnya tetap.
  - Isinya kini tiga zona menurut peringkat tambahan absolut (REG_263, REG_045, REG_270), dengan dua seri
    berdampingan: COP 3 (jingga) dan COP 2,5–3,5 per iklim (abu-abu). Satuannya GWh, sama dengan *slide* 9.
  - Kolom peringkat ("COP 3 → iklim") mengikuti grafik. Dua belas kotak label lama dipakai ulang dan lima dihapus.
  - Kotak INTERPRETASI berisi satu kalimat: "peringkat tambahan absolut stabil di kedua skenario COP (REG_263 1→1,
    REG_045 3→3, REG_270 4→4), sedangkan peringkat persen tidak (REG_290 1→47): titik awal menurut nilai absolut".
    Frasa "sehingga … dipilih dari" dari prompt dipadatkan agar muat empat baris pada 14 pt.
  - Karena grafik persen hilang, teks rantai lain ikut disesuaikan:
    - TEMUAN kini menyebut "lima zona dengan kenaikan persen tertinggi", supaya tidak terbaca sebagai zona di grafik
      GWh.
    - REKOMENDASI mempertahankan batasan "energi tahunan", yang sebelumnya ada di kotak interpretasi v4.
    - LANGKAH memakai urutan peringkat REG_263, REG_045, REG_270.
  - Jarak antarkotak rantai dirapatkan; kotak LANGKAH dan Pelaksana tetap di posisi v4.
- **Slide 11 (B).**
  - TEMUAN memakai opsi 1, "9,46% (gas jaringan 60–80%) vs 2,73% (>80%, 70% kawasan)". Opsi ini muat dalam tiga baris
    tanpa menambah kata, dan membandingkan dua kelompok besar (26% dan 70% kawasan latih). Pita ≤40% hanya berisi
    0,55% kawasan.
  - REKOMENDASI mendapat kalimat "Q2 dipilih: intensitas gas di atas median, kenaikan listrik tahunan (%) di bawah
    median". Kalimat ini juga menjawab mengapa Q2 dan bukan Q4, karena kenaikan listrik Q4 lebih rendah lagi.
  - Kotak LANGKAH yang dua baris dipendekkan dan diturunkan, sehingga kotak REKOMENDASI muat dua baris tambahan.
- **C2, *slide* 15 (F).**
  - Butir kedua ditambahkan: "Setelah koreksi dan penyesuaian akhir: skor *leaderboard* publik 0,24913 (rata-rata lima
    target)". Kata *leaderboard* dicetak miring seperti di *slide* lain.
  - Kutipan label "koreksi dan lainnya" di catatan C2 disamakan dengan label baru. Catatan juga menyebut bahwa galat
    per target setelah koreksi tidak dicetak di notebook.
- **C12, *slide* 25 (F).**
  - *Slide* baru setelah C11 dengan *layout* CADANGAN. Penanda, judul, dan nomor *slide* disalin dari C11. Tabelnya
    bergaya C9, tanpa baris sorotan, dengan selang-seling putih dan abu-abu muda seperti tabel cadangan lain.
  - C12 dimasukkan ke *section* "Cadangan tanya jawab", karena di dalam *deck* tidak ada daftar *slide* cadangan lain.
  - Judul: "Tidak ada pencilan yang dibuang". Tabel berisi nilai hilang, pencilan, batas data, dan target log. Alasan
    target log ditulis "[PERLU DIISI: alasan log target]".
  - Teks di layar berjumlah 59 kata, termasuk penanda, judul, dan nomor *slide*. Isian alasan nanti sebaiknya tidak
    lebih dari 6 kata supaya tetap di bawah 60.
  - Catatan memuat tiga pertanyaan juri dan empat butir pedoman jawaban. Penjawabnya Orang 2 (teknis), dengan nama
    "[PERLU DIISI]".
- **Catatan *slide* 1 sampai 13 (H).** Narasi ditulis ulang, sedangkan baris waktu dan peran tidak diubah. Kata
  dihitung dengan pemisah spasi, hanya pada paragraf narasi. Baris waktu, baris "Bila ditanya" (*slide* 5 dan 9), dan
  baris "Sumber" tidak dihitung.
  - Alur SHAP → rekomendasi → pelaksana tetap ada di *slide* 9 sampai 11. Frasa "Temuan model mendukung rekomendasi
    ini" dipertahankan.
  - Batasan klaim dipertahankan: asosiasi, bukan sebab-akibat; energi tahunan, bukan beban puncak; daftar skrining;
    indikator.
  - Kalimat serah terima ada di *slide* 2, 7, dan 11.
  - *Slide* 6 ditulis satu kalimat lisan per fakta.

## Tugas yang dilewati atau diubah, beserta alasannya

1. **C.3 dilewati.** Tidak ada kode yang menegakkan gas = EUI × luas lantai pada prediksi akhir.
   - `pipeline._assemble` (In [13]) menggabungkan kolom gas dengan bobot yang menekankan kawasan besar, dan kolom EUI
     tanpa penekanan itu. Koreksi galat dan penyesuaian akhir juga dihitung per kolom.
   - Rasio gas/(EUI × luas) di submisi berkisar 0,78 sampai 1,08 (median 1,0002).
2. **B.1 tidak memakai "paling kecil" dan "tekanan ke gardu".**
   - Rata-rata kenaikan Q4 (+87,04%) lebih rendah daripada Q2 (+90,24%) (In [34]).
   - Data hanya memuat energi tahunan, tanpa beban puncak atau kapasitas gardu (bagian 7.2).
   - Dalam tambahan absolut, zona Q2 tidak selalu kecil: REG_033 menambah 185 GWh, peringkat 52 dari 330.
   - Kalimat yang dipakai menyebut kedua sumbu kuadran, dan berlaku untuk setiap zona Q2.
3. **E tidak memakai kata "terbesar".** Klaim itu tertulis di notebook, tetapi ablasi pengembangan tidak mendukungnya.
   Frasa layar ditulis "perbaikan galat EUI model tunggal dari kelompok fitur ini".
4. **F, alasan target log.**
   - Notebook tidak menulis mengapa sebagian anggota menebak log. Kalimat di atas In [23] hanya menjelaskan skala
     grafik.
   - Skewness (gas 1,22 menjadi −0,29 sesudah log10) dihitung di luar notebook, dan *writeup* vB 4.2 menyebut
     keragaman pola galat. Keduanya bukan output notebook, jadi alasannya ditandai [PERLU DIISI].
5. **F, galat setelah koreksi.** Galat per target setelah koreksi tidak dicetak notebook maupun `summary.json`.
   - Yang tersedia adalah skor *leaderboard* agregat 0,24913 (In [28]), dan angka itu yang dipakai di C2.
   - Angka per target hanya ada di log pengembangan (*holdout*, sebelum penyesuaian akhir), dan tidak dipakai.
6. **A.3.** Catatan C10 memuat kriteria "titik awal dipilih dari zona yang tetap sepuluh besar dalam persen dan
   tambahan absolut pada kedua skenario, … ditambah REG_263 sebagai tambahan absolut terbesar".
   - Kriteria itu menyisihkan REG_273, peringkat 2 tambahan absolut, karena peringkat persennya 47 dan 32.
   - Namun REG_263 (peringkat persen 25 dan 51) juga hanya lolos lewat pengecualian, dan kriteria ini berbeda dari
     kalimat interpretasi v5 yang memakai nilai absolut.
   - Alasan khusus untuk REG_273 tidak tertulis, jadi tidak dikarang dan dicatat sebagai pertanyaan juri.
7. **G.** Nama pembicara belum ada, jadi diisi penanda.
8. **D, bentuk baris kaki.**
   - Rujukan "bagian 2.1/2.2" diganti "bagian 2" (*slide* 3, 4, dan 5), supaya tetap benar di notebook terkirim.
   - Baris kaki *slide* lain di luar tugas tetap memakai awalan "Notebook:", sehingga awalannya kini campur ("Sumber:"
     di *slide* 3, 4, 5, 8 sampai 11).

## Jumlah kata catatan *slide* utama

| Slide | Detik | Kata v4 | Kata/menit v4 | Kata v5 | Kata/menit v5 |
|---|---|---|---|---|---|
| 1 | 25 | 50 | 120,0 | 44 | 105,6 |
| 2 | 35 | 73 | 125,1 | 61 | 104,6 |
| 3 | 55 | 120 | 130,9 | 95 | 103,6 |
| 4 | 50 | 98 | 117,6 | 86 | 103,2 |
| 5 | 45 | 98 | 130,7 | 79 | 105,3 |
| 6 | 60 | 127 | 127,0 | 106 | 106,0 |
| 7 | 30 | 58 | 116,0 | 52 | 104,0 |
| 8 | 55 | 107 | 116,7 | 93 | 101,5 |
| 9 | 50 | 93 | 111,6 | 83 | 99,6 |
| 10 | 50 | 95 | 114,0 | 89 | 106,8 |
| 11 | 55 | 116 | 126,5 | 95 | 103,6 |
| 12 | 35 | 76 | 130,3 | 60 | 102,9 |
| 13 | 25 | 56 | 134,4 | 44 | 105,6 |
| **Total** | 570 | **1.167** | 122,8 | **987** | 103,9 |

Tidak ada *slide* di atas 110 kata per menit.

## [PERLU DIISI] yang tersisa

- *Slide* 1: [PERLU DIISI: nama dan peran pembicara].
- C12 (*slide* 25), layar dan catatan: [PERLU DIISI: alasan log target].
- C12 (*slide* 25), catatan: [PERLU DIISI: nama Orang 2 sesuai slide 1].

## Angka yang berbeda dari v4

| Slide | v4 | v5 |
|---|---|---|
| 5 | "galat EUI model tunggal −0,0022" | angka dipindah ke catatan |
| 10 (grafik) | +121,41%, +115,10%, +114,29%, +113,91%, +113,46%, median +94,85%; peringkat COP per iklim 47, 6, 7, 8, 1 | 757/805, 442/525, 434/514 GWh (COP 3 / COP 2,5–3,5 per iklim); peringkat 1→1, 3→3, 4→4 |
| 10 (interpretasi) | tanpa angka | REG_290 1→47 dan peringkat tambahan absolut |
| 11 | 40,24% (gas jaringan ≤40%) | 9,46% (gas jaringan 60–80%) |
| 6 | (tidak ada) | 56 sel, 17 modul, ±5 jam |
| C2 | (tidak ada) | 0,24913 setelah koreksi dan penyesuaian akhir |
| catatan 1, 3, 5, 11 | "hampir 95 persen", "sekitar tiga kali", "lebih dari 42 ribu", "sekitar 40 persen" | 94,85; 2,8 dan 3,0; 42.931; 9,46, disamakan dengan layar |
| catatan 8, 10, 11, 13 | 613 ribu kawasan; 113 sampai 121 persen; 100 persen; "hampir dua kali lipat" | dihapus dari narasi karena tidak tertulis di *slide* yang sama |

Tidak ada angka yang muncul sama persis di *slide* atau catatan lain sehingga perlu diganti mengikuti aturan 4.
"40,24" hanya ada di *slide* 11, sedangkan "−0,0022" hanya ada di *slide* 5 beserta catatannya.

## Angka dari prompt yang belum terverifikasi

Tidak ada. Seluruh angka di prompt terverifikasi ke output notebook atau skrip cek: 1→1, 3→3, 4→4, 1→47, 40,24%,
2,73%, 9,46%, 0,55%, 56 sel, 17 modul, ±5 jam, ±37/±173 m², 96,5%, +65%, dan −0,0022. Yang tidak terbukti adalah
tiga klaim kata, yaitu "terbesar", "paling kecil", dan "tekanan ke gardu".

## Pertanyaan juri yang perlu disiapkan

1. **REG_273 dan titik awal R2.** Mengapa REG_273, peringkat 2 tambahan absolut pada kedua skenario (616 → 707 GWh),
   tidak menjadi titik awal R2, padahal *slide* 10 memilih titik awal menurut nilai absolut?
   - Lihat butir A.3 di atas. Tim perlu memutuskan jawabannya: memakai kriteria C10 atau menyamakan C10 dengan *slide*
     10.
   - REG_273 sudah menjadi titik awal R1 karena potensi hemat bangunannya terbesar (In [35]).
2. **Peringkat tambahan absolut yang "stabil".** Pada COP 3 tambahan absolut = 0,2667 × gas zona, jadi peringkatnya
   sama dengan peringkat volume gas. Stabilitasnya sebagian karena konstruksi, dan yang tetap hanya peringkat teratas
   (REG_290 bergeser 46 → 60).
3. **REG_033 sebagai contoh Q2.** Tambahan absolutnya 185 GWh (peringkat 52 dari 330). Kenaikan yang di bawah median
   adalah kenaikan dalam persen.
4. **Klaim "terbesar" di notebook.** Notebook bagian 2 masih menulis "perbaikan terbesar", sedangkan *slide* 5 tidak
   memakai kata itu. Jawaban ada di baris "Bila ditanya" catatan *slide* 5.
5. **Atribusi 0,0007 di notebook.** Sel ke-40 menulis koreksi galat wilayah menurunkan skor 0,0007 (v12 ke v15),
   padahal langkah itu memuat lima perubahan dan bagian kNN sekitar 0,0004.
6. **Gas dan EUI pada prediksi.** Identitas gas = EUI × luas lantai tidak ditegakkan pada prediksi (rasio 0,78 sampai
   1,08).
7. **Galat per target setelah koreksi.** Notebook tidak mencetaknya. Yang tersedia hanya skor agregat 0,24913.
8. **Notebook 7.3 butir 2 dan 4.** Butir 2 menulis "zona lain", padahal Q4 lebih rendah. Butir 4 menulis "hampir
   hanya", padahal 82,92% rumah tangga berpemanas listrik tinggal di kawasan dengan gas jaringan >60%.
9. **+65% dan 66,5%.** Jawaban ada di baris "Bila ditanya" catatan *slide* 9. C11 menulis "pita 5% ekstrem … +65%",
   sedangkan nilai skripnya 66,5%.
10. **Label *slide* 7 yang juga gabungan.** "objektif baru" juga memuat FB07b, dan "model NCA" juga memuat perluasan
    konfigurasi kNN EUI.
11. **Mode tanpa pelatihan.** Mode ±5 jam butuh prediksi tersimpan yang tidak ikut dikirim. Pelatihan penuh sekitar
    20 jam.
12. **Dua versi notebook.** Notebook terkirim (23 Okt) belum memuat subjudul 2.1 sampai 2.3 dan blok "Cara membaca
    diagram". Nomor In [n] dan outputnya sama.

## Validasi

**Struktur.** Pemeriksaan pertama dijalankan sebelum tinjauan, lalu diulang sekali setelah satu putaran perbaikan,
tanpa render ulang.
- v5 terbuka dengan python-pptx dan berisi 25 *slide* (v4 24 + 1). Sidik sha256 v4 tidak berubah.
- Semua teks yang ditambah atau diubah berukuran 12 pt ke atas dan punya ukuran eksplisit; teks isi 14 sampai 17 pt,
  baris kaki 12 pt.
- Tidak ada em dash, dan tidak ada objek di luar *slide*.
- Keenam *section* lengkap ("Cadangan tanya jawab" kini 12 *slide*).
- Naskah 987 kata (103,9 kata per menit), semua *slide* di bawah 110 kata per menit, dan setiap angka yang diucapkan
  ada di *slide* yang sama.
- `validate.py` skill pptx (dengan v4 sebagai acuan) lulus.

**Visual.**
- LibreOffice tidak tersedia, jadi *slide* yang berubah (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 25) dirender sekali
  dengan perender Carlito `deck/ppt03/render_pptx.py`. Carlito memiliki metrik yang sama dengan Calibri.
- Hasilnya: tidak ada teks meluap, kata terpotong, tabel memanjang, atau objek keluar *slide*.
- Tiga peninjau berkonteks bersih (visual, kepatuhan arahan, dan bahasa) memeriksa render dan teks. Mereka menemukan 3
  temuan wajib dan sejumlah saran. Perbaikannya diterapkan sekali:
  - baris kaki *slide* 5 yang merujuk "bagian 2.2";
  - narasi *slide* 6 yang terdengar seolah dua jalur digabung jadi satu;
  - judul C12 yang memuat dua pesan, dengan jumlah kata yang memperhitungkan penanda dan nomor *slide*;
  - label pembicara di *slide* 1;
  - "salin duplikat" di *slide* 7;
  - legenda COP, TEMUAN, REKOMENDASI, dan urutan LANGKAH di *slide* 10;
  - alasan Q2 di *slide* 11;
  - baris sorotan dan tinggi tabel C12;
  - *leaderboard* miring di C2;
  - beberapa kalimat naskah.
- Kecocokan teks dalam kotak setelah perbaikan dicek dengan pengukur lebar Carlito: semua teks muat. Render tidak
  diulang, sesuai arahan.
- Tampilan akhir tetap perlu dilihat sekali di PowerPoint, terutama grafik dua seri di *slide* 10. Untuk sumbu
  kategori terbalik, perender menaruh seri pertama di atas, sama dengan perilaku PowerPoint.

## Catatan di luar tugas (tidak diubah)

- **C10 (*slide* 23), catatan.** Masih menulis "urutan program di Q2 mengikuti tekanan jaringan". Kalimat ini tidak
  diubah karena C10 di luar tugas. Usulan penggantinya: "mengikuti kenaikan listrik tahunan (persen); beban puncak dan
  kapasitas gardu tidak ada di data".
- **C6 (*slide* 19).** Masih menulis urutan REG_263, REG_270, REG_045. LANGKAH *slide* 10 kini memakai urutan
  peringkat.
- **Skrip `cek_kebijakan.py`.** Skrip membaca `reports/policy/shap_*.npz`, artefak pengembangan yang masuk
  `.gitignore`, sehingga tidak bisa dijalankan dari *clone* bersih. Nilainya konsisten dengan In [32].
- **Dokumen struktur.** `deck/ppt03/01_struktur_deck_v4.md` dan pembangun v4 tidak diperbarui, karena v5 disunting
  langsung pada PPTX sesuai arahan.
