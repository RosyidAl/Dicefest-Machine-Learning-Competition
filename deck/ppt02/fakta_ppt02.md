# Fakta Tambahan untuk *Deck* v3 (ppt02)

Lembar fakta ini melengkapi `../fakta_notebook.txt`, `../fakta_turunan.md`, `../ppt01/fakta_tambahan.md`, dan
*writeup* B (`reports/WRITEUP_DRAFT_vB.md`). Isinya hanya angka di `01_struktur_deck_minto.md` yang tidak tercetak
persis di keempat berkas tersebut. Kelima berkas dipakai bersama oleh `cek_struktur.py` dan `check_deck.py`. Nomor sel
dihitung dari 0.

## A. Jumlah model dan nilai SHAP

| Angka | Dipakai di | Asal | Cara mendapatkan |
|---|---|---|---|
| 43 model | *slide* 6; A4; C4; E | sel 23 (28 di gabungan gas/EUI dan 13 di gabungan listrik) dan sel 35 (dua LightGBM adopsi, `q_xent` dan `r_reg`) | 28 + 13 + 2 |
| 16, 5, 2, 3, 1, 1 (gas/EUI) dan 3, 4, 1, 2, 2, 1 (listrik) | C4 | tabel anggota sel 31 | jumlah baris per keluarga model dan jalur; sama dengan tabel `reports/KAGGLE_WRITEUP_TOP20.md` bagian 3.3 |
| 550 model terlatih | C4 | `reports/KAGGLE_WRITEUP_TOP20.md` bagian 3.3 | seluruh anggota dilatih per *fold*, sebagian dengan tiga *seed* |
| 0,08953 dan 0,08948 | C5 | `reports/policy/shap_eui.npz` dan `shap_epm.npz`, yaitu nilai SHAP yang sama dengan sel 45 (dicek oleh `../ppt01/make_visuals_v2.py`) | rata-rata nilai mutlak SHAP `area_per_meter` dengan presisi penuh: 0,089530 dan 0,089476; sel 45 mencetak keduanya sebagai 0,0895 |
| 4 model penjelas | C5; E | sel 41 (empat target model penjelas: log EUI, log gas per meter, log listrik per meter, adopsi) | jumlah baris tabel R² |
| 0,0814 (porsi kawasan uji di subzona baru) | *slide* 6, baris Sumber di catatan pembicara | sel 35 (`unseen_fraction` 0.08141336) | dibulatkan empat desimal; di layar ditulis 8,14% |

## B. Uji kebocoran baris latih

| Angka | Dipakai di | Asal | Keterangan |
|---|---|---|---|
| 309.097 baris bagian lain berubah | *slide* 5; C3; G2 | `../ppt01/bukti/uji_kebocoran_oof_dalam.json` (`berubah_bagian_lain`) | baris latih di empat bagian dalam lainnya; berubah karena memakai bagian 0 sebagai tetangga. Syarat lulus uji (`uji_kebocoran_oof_dalam.py`) |

## C. Zona contoh dan urutan subzona (`bukti/cek_zona_contoh.py`, `bukti/cek_zona_contoh.json`)

Skrip membangun data zona dengan cara yang sama dengan `../ppt01/bukti/cek_kebijakan.py` dan memeriksa ulang angkanya
dengan *assertion*: 141,35 TWh, median 94,85%, jumlah zona per kuadran 118/47/47/118, REG_249 85,54% dan 3,75%
(sel 50), serta Spearman 0,8147 dan 68 zona pindah kuadran pada COP per iklim.

| Angka | Dipakai di | Keterangan |
|---|---|---|
| REG_290: kenaikan relatif peringkat 1 (COP 3) dan 47 (COP per iklim); tambahan absolut peringkat 46 dan 60 | A2; C10; E14 | tetap Q1 pada kedua skenario |
| REG_270: kenaikan relatif peringkat 5 dan 1; tambahan absolut peringkat 4 pada keduanya | *slide* 10; C10 | Q1 pada kedua skenario |
| REG_045: kenaikan relatif peringkat 9 dan 3; tambahan absolut peringkat 3 pada keduanya | *slide* 10; C10 | Q1 pada kedua skenario |
| REG_263: tambahan absolut peringkat 1 pada keduanya; kenaikan relatif peringkat 25 dan 51 | *slide* 10; C10 | Q1 pada kedua skenario |
| REG_273: tambahan absolut peringkat 2 pada keduanya | C10 | Q1 pada kedua skenario |
| REG_259 dan REG_275: kenaikan relatif peringkat 2 dan 3 (COP 3) serta 6 dan 7 (COP per iklim); tambahan absolut peringkat 25 dan 53 (COP 3) serta 17 dan 41 (COP per iklim) | C10 | Q1 pada kedua skenario; tinggi dalam persen, kecil dalam tambahan absolut |
| enam zona tetap sepuluh besar kenaikan relatif pada kedua skenario (REG_045, REG_259, REG_260, REG_270, REG_275, REG_330); hanya REG_270 dan REG_045 yang juga sepuluh besar tambahan absolut pada kedua skenario | A3; *slide* 10; C10 | dihitung ulang oleh `make_visuals_v3.py` (`img/data_grafik.json`, kunci `s10_kenaikan`) |
| REG_249: +90,07% (Q2) menjadi +95,64% (Q1); median COP per iklim 94,60% | C10; E14 | pindah kuadran |
| REG_033: gas jaringan 83,21%; adopsi 4,99%; kenaikan +84,92% (COP 3) dan +78,37% (COP per iklim) | *slide* 11; C6; C10 | Q2 pada kedua skenario; gas 0,695 TWh, terbesar ketiga di antara Q2 yang tetap Q2 |
| 33 dari 47 zona Q2 dan 98 dari 118 zona Q1 tetap di kuadrannya | C10 | COP 3 seragam dibandingkan dengan COP per iklim |
| 96,5% pasangan subzona bernomor berurutan di zona yang sama; 163.185 subzona | *slide* 4; A4; E12 | subzona unik diurutkan menurut nomor |
| 73,39% pasangan baris berurutan di subzona yang sama | E12 | baris diurutkan menurut nomor subzona, seperti `report.split_checks`; 99,07% per baris tereproduksi |

## D. Angka turunan dari keluaran *notebook*

| Angka | Dipakai di | Asal | Cara mendapatkan |
|---|---|---|---|
| Q1 5,5% dan Q2 6,6% (indikator potensi *retrofit* relatif terhadap konsumsi gas kuadran) | C10 | sel 49: Q1 3,650 / 66,095 TWh; Q2 1,073 / 16,299 TWh | 0,0552 dan 0,0658, dibulatkan |
| hampir separuh potensi *retrofit* di Q1 | A3 (dalam kata) | sel 49 (3,650 TWh) dan sel 50 (7,39 TWh) | 3,650 / 7,39 = 0,494 |

## E. Angka pada grafik asli PowerPoint (`img/data_grafik.json`, ditulis `make_visuals_v3.py`)

Grafik batang di PPTX v3 dibaca dari berkas ini. Setiap angka diperiksa dengan *assertion* terhadap sumber yang
tercantum sebelum berkas ditulis.

| Grafik | Angka | Asal |
|---|---|---|
| *slide* 3: lima fitur teratas | log EUI: 0,0895; 0,0321; 0,0276; 0,0260; 0,0250. Listrik per meter: 0,0895; 0,0294; 0,0222; 0,0196; 0,0163 | sel 45 (`reports/policy/shap_eui.npz`, `shap_epm.npz`) |
| *slide* 9: indikator potensi *retrofit* tiga zona | REG_273 170,1 GWh (7,36%); REG_270 139,9 GWh (8,60%); REG_045 110,8 GWh (6,69%) | sel 50 tabel pertama: gas × saving_pct |
| *slide* 10: kenaikan relatif tertinggi | REG_290 +121,41% (peringkat 47 pada COP per iklim); REG_259 +115,10% (6); REG_275 +114,29% (7); REG_260 +113,91% (8); REG_270 +113,46% (1); median +94,85% | sel 50 tabel kedua; peringkat COP per iklim dihitung ulang dan dicocokkan dengan `bukti/cek_zona_contoh.json` |
| *slide* 11: tingkat adopsi per porsi gas jaringan | ≤40%: 40,24% (0,55% kawasan); 40–60%: 26,92% (3,04%); 60–80%: 9,46% (26,32%); >80%: 2,73% (70,08%) | baris latih dibobot jumlah meter; sama dengan `../ppt01/bukti/cek_kebijakan.json` |
| C5: porsi kontribusi per kelompok fitur | *writeup* B Tabel 3 | dibaca langsung dari `reports/WRITEUP_DRAFT_vB.md` |
