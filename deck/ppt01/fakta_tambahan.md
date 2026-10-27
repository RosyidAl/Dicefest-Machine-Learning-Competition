# Fakta Tambahan untuk Deck Final v2

Lembar fakta ini melengkapi `../fakta_notebook.txt` (teks dan keluaran semua sel notebook) dan `../fakta_turunan.md`
(angka hasil pembulatan atau nilai kode). Isinya adalah angka di deck v2 yang tidak tercetak persis di notebook. Ketiga
berkas dipakai bersama oleh `check_deck.py`. Nomor sel dihitung dari 0.

## A. Hasil uji baru (dijalankan 27 Okt 2026 dengan modul notebook yang sama)

| Angka di deck | Slide | Sumber | Keterangan |
|---|---|---|---|
| 0 baris berubah; 42.931 baris validasi; 183.987 baris uji; 386.372 baris latih | 1, 5, 14, C7 | `bukti/uji_kebocoran_notebook.json` | label kelima target pada baris validasi fold 0 dari KFold(10, shuffle, seed 42) diacak; enam blok fitur bertarget identik bit |
| 39 kolom kNN-target, 42 kolom rata-rata wilayah | C7 | idem | FB07 + FB07b (`fb07b.DataX`, `use_x=True`) dan `te_features(windows=(2, 10), cells=True)` |
| 42.931 baris berubah pada kontrol (15 kolom) | 5, C7 | idem | kontrol sengaja bocor: tetangga diambil dari semua baris latih, termasuk baris validasi |
| 1207 detik | catatan | idem | durasi uji |
| skor jujur global gas 0,24494; EUI 0,65023; listrik 0,31587 | C1 | `bukti/skor_jujur_bobot_adaptif.json` | sama persis dengan sel 35 (dicek otomatis) |
| skor jujur adaptif gas 0,24480; EUI 0,64978; listrik 0,31585 | C1 | idem | 0,2447961; 0,6497756; 0,3158481; `blend.blend_vc` di-fit ulang di setiap fold luar |
| selisih −0,000146; −0,000453; −0,000027; rata-rata 5 target −0,000125 | C1 | idem | −0,00014563; −0,00045294; −0,00002652; −0,00012502 |

## B. Skor dan peringkat dari Kaggle

| Angka | Slide | Sumber |
|---|---|---|
| 0,24844, peringkat 8 (LB private) | 1, 7, 14 | Kaggle LB private, ditempel user 24 Okt 2026 00:08 WIB (`reports/STATUS.md`) |
| 0,24913 (LB publik) | 7, 14 | Kaggle LB publik; juga sel 37 |
| 24 Okt 2026 | 7 | tanggal LB private |

## C. Angka turunan dari keluaran notebook

| Angka di deck | Slide | Asal | Cara mendapatkan |
|---|---|---|---|
| 2,8× dan 3,0× | 4 | sel 45: 0,0895 / 0,0321 dan 0,0895 / 0,0294 | 2,79 dan 3,04, dibulatkan |
| ≈170; ≈140; ≈111; ≈101; ≈97 GWh | 10 | sel 50 tabel pertama: gas × saving_pct | 170,1; 139,9; 110,8; 101,3; 97,4 GWh (`img/angka_visual.json`) |
| 47%; 12%; 15%; 27% | 1, 9 | sel 49 `porsi_gas` 0,468; 0,115; 0,146; 0,271 | dikali 100, dibulatkan |
| +102%; +90%; +99%; +87% | 9 | sel 49 `kenaikan_beban_rata2_pct` 102,494; 90,245; 99,102; 87,042 | dibulatkan |
| 1,07; 0,86; 1,81 TWh | 9 | sel 49 `hemat_retrofit_TWh` 1,073; 0,858; 1,810 | dibulatkan |
| 113% sampai 121% | 11 | sel 50 tabel kedua: 113,46 sampai 121,41 | dibulatkan |
| R² 0,49 sampai 0,75 | 13 | sel 41: 0,491 dan 0,747 | dibulatkan |
| median 94,85%; rata-rata 94,74%; terendah 67,67%; tertinggi 121,41% | 2, 9, C4 | data zona sel 49 dihitung ulang (`make_visuals_v2.py`, cocok dengan sel 49 dan 50) | `img/angka_visual.json` |
| +99,6%; +83,0%; +71,1%; +113,8%; +94,8%; +81,3%; +128,0%; +106,7%; +91,5% | C4 | median 94,85% × (porsi / 0,8) × (3 / COP) | `img/angka_visual.json`; dalam bilangan bulat sama dengan *writeup* Tabel 4 (+100%, +83%, +71%, +114%, +95%, +81%, +128%, +107%, +92%) |
| peringkat zona identik untuk seluruh 330 zona | 9, C4 | peringkat kenaikan pada (80%, COP 3) vs (70%, COP 2,5) | `peringkat_tetap: true` di `img/angka_visual.json` |
| ±30 m² dan ±300 m² | 8 | teks sel 48 (sekitar 30 m² dan 300 m²) | ujung garis rata-rata: 27,1 dan 310,3 m² |
| 20.000 baris | 4, C8 | `explain.shap_values(n=20000, seed=0)` | ukuran sampel SHAP |

## D. Angka dari *writeup* dan catatan pengembangan

| Angka | Slide | Sumber |
|---|---|---|
| 0,2573; 0,6721; 0,3262 / 0,2562; 0,6699; 0,3255 / 0,2708; 0,6923; 0,3302 / 0,3637; 0,8331; 0,3474 | C5 | *writeup* Tabel 1 (LightGBM, skema acak per baris dan per zona, tanpa dan dengan rata-rata target wilayah) |
| efisiensi ketel lama 88%, nilai turun sekitar 12% | C4 | *writeup* bab 7.3 |
| 0,25095 vs 0,25096 | C7 | `reports/DECISIONS.md` D002: bobot NNLS dari prediksi *out-of-fold* vs dari *holdout* P70/H30 |
| sekitar 40% (sosiodemografi pada listrik per meter) | C8 | grafik sel 43, dibaca dari grafik (`../02_naskah_kebijakan.md` bagian C) |

## E. Nilai kode dan konfigurasi (slide cadangan)

| Angka | Slide | Sumber |
|---|---|---|
| kNN full W 100 k 20; era W 30 k 20; fc W 300 k 50; full W 10 k 10; era W 300 k 50; full W 3000 k 200 | C2 | `configs/resid_knn.json` kunci `configs` → `gas` (ditulis sel 22) |
| 800 putaran LightGBM model penjelas | C8 | `explain.fit_explain(rounds=800)` (modul di sel 19) |
| −0,000124 (validasi bersarang A2xM2xp2 saat pengembangan) | C1 (catatan) | `reports/DECISIONS.md` D012 |

## F. Pemeriksaan setelah audit independen (27 Okt 2026, `bukti/cek_kebijakan.py` dan `.json`)

| Angka di deck | Slide | Keterangan |
|---|---|---|
| 184 ribu kawasan diprediksi | 1 | 183.987 baris uji (sel 25); 613 ribu = 429.303 latih + 183.987 uji |
| tingkat adopsi 40,24%; 26,92%; 9,46%; 2,73% | 12, Q&A | baris latih, dibobot jumlah meter, menurut porsi gas jaringan ≤40%, 40-60%, 60-80%, >80% |
| porsi kawasan 0,55%; 3,04%; 26,32%; 70,08% | 12, Q&A | idem; slide menulis "70% kawasan" untuk >80% |
| porsi rumah tangga berpemanas listrik 3,42%; 13,66%; 47,04%; 35,88% | Q&A | idem; mayoritas pengguna pemanas listrik justru tinggal di kawasan bergas 60% ke atas |
| 254 dari 330 zona di atas 80%; median zona 82,42% | Q&A | porsi gas jaringan zona dibobot jumlah meter |
| Spearman 0,998 (0,9981; terendah 0,9974); median 8 zona pindah kuadran (maks 14); 5 zona teratas tetap | 9, C4 | galat prediksi pada baris uji: derau log-normal 0,161 (gas) dan 0,206 (listrik), 200 simulasi |
| Spearman 0,81 (0,8147); 68 zona pindah kuadran; 1 dari 5 zona teratas tetap | 9, 11, 13, C4 | COP 3,5 di zona beriklim teringan sampai 2,5 di terberat, menurut rata-rata `env_climate_05` zona |
| Spearman 0,8147; median 72 zona pindah kuadran | Q&A | derau dan COP per iklim sekaligus |
| REG_263 0,76 TWh (0,757); REG_273 0,616; REG_045 0,442; REG_270 0,434; REG_056 0,391 TWh | 11, Q&A | tambahan listrik absolut = gas × 0,8 / 3; REG_290 peringkat 46; Spearman persen vs absolut 0,36 |
| sekitar separuh (0,513); 0,643 | 13, Q&A | porsi potensi *retrofit* bila hanya periode bangunan dihitung; bila fitur ukuran dan jumlah dikeluarkan (sampel SHAP 20.000 baris) |
| ±37 m² (36,6) dan ±173 m² (173,1) | 8 | rata-rata luas per meter pada 5% kawasan terkecil dan terbesar (pita qcut 20 di sel 47) |
| +66,5% gas per m²; +74,6% listrik per rumah | 8 | kontras pita 5% ekstrem; teks sel 48 membulatkan menjadi sekitar 65% dan 75% |
| +34% (33,7) dan +35% (34,6); kuartil 61,1 dan 94,8 m² | 8 | kontras kuartil bawah vs atas |

## G. Uji baris latih (`bukti/uji_kebocoran_oof_dalam.py` dan `.json`)

| Angka | Slide | Keterangan |
|---|---|---|
| 77.275 baris latih bagian dalam 0; 0 baris berubah | 5, C7 | label kelima target pada bagian dalam 0 (KFold(5, shuffle, 42) atas 386.372 baris latih fold 0) diacak; kNN-target 39 kolom dan rata-rata wilayah 42 kolom identik bit |

## H. Riwayat *leaderboard* yang dipakai di tanya jawab

| Angka | Slide | Sumber |
|---|---|---|
| 0,2499 menjadi 0,24954 (koreksi kNN sekitar −0,0004) | C2 | `reports/JOURNAL.md`: daftar LB publik 15 Okt (0,2499) dan LB 16 Okt (0,24954) |
| 0,24938 vs 0,24934 (adopsi v3 ditolak karena LB) | Q&A | `reports/DECISIONS.md` D011 |
| notebook 56 sel | 6 | `../01_naskah_fitur_arsitektur.md` (notebook tereksekusi 56 sel); 17 modul tertulis di sel 3 |
| −0,00009 (bobot adaptif di *leaderboard*) | C1 | sel 37: v19 0,24934 menjadi v19vc 0,24925 |
