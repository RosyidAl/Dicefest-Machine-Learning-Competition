# EDA dan forensik data (Fase 1)

Sumber angka: `reports/eda/findings.json` (dibangkitkan `src/eda.py`) dan EDA lanjutan 12 sampai 15 Okt (scratchpad eda1 sampai eda15, leak1 sampai leak6, adoptN).
Figur ada di `reports/eda/figures/`.

## 5.A Integritas
- Semua fitur numerik kecuali dua kolom string:
  - `bld_dim_18`: empat kategori ordinal "High (<0%)" sampai "Very Low (>18%)", di-encode 0 sampai 3;
  - `socio_demog_016`: Urban/Rural, di-encode 1/0.
- NaN hanya di `bld_dim_04` dan `bld_dim_07`: 695 baris train dan 336 baris test, baris yang sama. Tidak ada inf.
  Satu nilai sentinel 9999 di `bld_dim_03`.
- 11 kolom konstan. Sebagian adalah duplikat persis: `bld_era_01`, `env_climate_06`, dan 8 kolom socio_demog.
  `env_climate_02` = `env_climate_04` persis.
  - Implikasi: dibuang (FB01, `common.constant_cols`).
- 6 pasangan |r| > 0,999, misalnya `socio_demog_032` = 100 - `socio_demog_033`, dan `bld_dim_13` hampir = `socio_demog_009`.
  Ada 75 pasangan |Spearman| > 0,95. Banyak redundansi, tetapi untuk GBDT tidak merugikan.
- Tidak ada baris duplikat persis (semua fitur + ID), baik di train, di test, maupun lintas train-test.
  Namun 77 baris test identik dengan baris train pada kunci (socio_demog_005, subzone, bld_dim_12, bld_dim_01, bld_dim_06).
  Di train, 92 grup dengan kunci ini selalu bertarget identik.
  - Implikasi: `dup_fix` menyalin target train untuk 77 baris itu (PP0003).
- 45 kolom bernilai integer penuh (count). Sebanyak 113 dari 132 kolom sensus (socio_demog, hvac_dist, env_climate) konstan di dalam subzone.
  Artinya kolom-kolom itu atribut Output Area (OA). Kolom lain (bld_dim, bld_era) adalah atribut baris (kode pos).

## 5.B Forensik split (D001)
| Bukti | Angka | Arti |
|---|---|---|
| Rasio baris train | 429.303 / 613.290 = 0,700000 | `train_test_split(test_size=0.3)` |
| id | train ID_1 sampai ID_429303 berurutan, test ID_429304 sampai ID_613290 | id hanya nomor urut setelah split |
| Spearman id vs target | \|rho\| <= 0,0026 untuk kelima target, 0,0007 untuk zone | id tidak membawa informasi |
| zone_region_id | 329 dari 330 zone ada di test, semuanya ada di train; 100% baris test zone-nya terlihat | |
| Rasio n_test/(n_train+n_test) per zone | rata-rata 0,2993; std z-score binomial 0,989 | persis sebaran binomial split acak |
| subzone_id | 153.508 di train, 110.548 di test, 100.871 beririsan; 91,86% baris test subzone-nya ada di train | |
| Baris test di subzone tak terlihat | teramati 8,14% vs harapan split acak per baris 8,17% | split acak per baris, bukan per grup |
| Hierarki | setiap subzone hanya di satu zone; 99,07% subzone bernomor berurutan berada di zone yang sama | `sub_n` berurut secara spasial |
| Adversarial validation (LGB 5-fold) | AUC 0,4997 tanpa ID, 0,4990 dengan ID | distribusi train = test |

Kesimpulan:
- Test adalah sampel acak 30% per baris.
- CV KFold acak per baris meniru test. Informasi tingkat grup (target encoding subzone/zone/window, agregat) sah.
- Kekhawatiran drift tidak ada, jadi adversarial reweighting (A03) tidak relevan.

Figur: `zone_test_ratio.png`, `id_rolling_mean.png` (rolling mean target vs id datar).

## 5.C Target
| Target | std train (ddof 1) | sigma beku | min | median | max | skew | Porsi 1% teratas pada jumlah kuadrat deviasi |
|---|---|---|---|---|---|---|---|
| t1 gas kWh | 151.750,04 | 151.750,04 | 4.324 | 193.549 | 999.765 | 1,22 | 14,4% |
| t2 elec kWh | 41.725,37 | 41.725,37 | 2.968 | 53.839 | 396.492 | 1,23 | 14,8% |
| t3 EUI | 40,652 | 40,65 | 5,1 | 134,8 | 648,1 | 1,30 | 20,2% |
| t4 clean % | 12,491 | 12,49 | 8,33 | 96,3 | 100 | -1,57 | 14,6% |
| t5 adopsi % | 7,185 | 7,18 | 0 | 3,10 | 93,2 | 3,40 | 34,9% |

Pengamatan:
- Sigma beku = std train ddof 1 (persis untuk t1, t2).
- Tidak ada nol kecuali t5 (33.859 nol). Tidak ada pelanggaran rentang 0 sampai 100.
- Gas terpotong di bawah 1e6 kWh (max 999.765).
- Korelasi Pearson: t1-t2 0,887; t1-t3 0,18; lainnya < 0,11.

Rata-rata grup OOF (5-fold acak) sebagai prediktor, NRMSE:

| Grup | gas | elec | EUI | clean | adopsi |
|---|---|---|---|---|---|
| zone | 0,972 | 0,984 | 0,936 | 0,910 | 0,968 |
| subzone | 1,113 | 1,145 | 1,032 | 0,949 | 0,566 |

- Rata-rata subzone kuat hanya untuk adopsi, yang memang konstan per subzone.
- Untuk target makro, sinyal grup baru berguna setelah dinormalisasi ukuran (target encoding rasio, misal EUI, kWh per meter).

Figur: `target_hist.png`.

## 5.D Fitur
- Blok komposisi (jumlah baris konstan 100) di socio_demog.
  - Contoh: 026 sampai 031, 034 sampai 038, 040 sampai 045, 046 sampai 054, 055 sampai 060, 061 sampai 069, 089 sampai 098, 099 sampai 106.
  - Ini share kategori sensus (tenure, tipe hunian, dst).
  - hvac_dist tidak berjumlah 100. Median 96,1 dan sisanya adalah kategori pemanas yang tidak dicantumkan.
    Dari sini didapat jumlah rumah tangga N eksak (adopt_v2).
- Level spasial:
  - env_climate paling konstan di dalam zone (rasio variansi dalam-zone 0,06 sampai 0,18).
  - env_climate_01/03/05 mendefinisikan sel grid iklim, sekitar 15 baris train per sel.
- Relasi univariat:
  - gas dan elec ditentukan ukuran (Spearman bld_dim_06 0,92 untuk gas, bld_dim_12 0,90 untuk elec).
  - EUI lemah terhadap fitur tunggal: Spearman maksimum 0,18 (bld_dim_32, bld_dim_20, socio_demog_082, env_climate_05).
  - clean = socio_demog_011 (Spearman 1,0).
  - Adopsi ditentukan hvac_dist_02 dan hvac_dist_09 (gain LGB 0,69 dan 0,15).

## 5.E Struktur fisik
| Relasi | Hasil | Implikasi |
|---|---|---|
| gas = EUI x bld_dim_06 | log-std rasio 0,000; selisih relatif maks 6,7e-16 | bld_dim_06 = luas lantai. Satu model gas sekaligus memberi EUI (T10); bobot blend dipisah per metrik |
| exposure elec | log-std elec/bld_dim_12 0,318 (terbaik) | bld_dim_12 = jumlah meter; formulasi elec per meter (epm, lepm) |
| clean = round(100 bld_dim_02 / bld_dim_01, 2) | eksak di 100% baris | t4 diprediksi eksak (NRMSE 0) |
| adopsi konstan per subzone | ya, di semua subzone | baris di subzone terlihat disalin; hanya sekitar 8% baris test perlu model |
| adopsi vs hvac_dist | R2 linear 0,978; adopsi <= sisa hvac di 100% baris | adopsi = sisa hvac - satu kategori hilang (model binomial q) |
| urutan bld_era | bld_era_02..08 dan 09..15 adalah dua blok paralel dengan pola korelasi berulang vs EUI | era tua dan baru berkorelasi positif dengan EUI, era tengah negatif |

## 5.F Temuan terpenting dan implikasinya
1. Split acak per baris (D001): KFold acak jadi CV utama; TE subzone/zone/window/sel iklim sah. Sudah terpakai di semua model.
2. Tiga identitas eksak (gas = EUI x A, clean = rasio bld_dim, adopsi konstan per subzone): dua dari lima target praktis selesai, dan EUI serta gas berbagi satu model.
3. Target makro berskala ukuran. Prediksi rasio per exposure (meter, luas, jumlah EPC, GEA) dengan bobot E^2 menyamakan loss dengan MSE skala asli (T4).
4. Noise label besar: varians residual log sekitar a + b/M (rata-rata rumah tangga), sekitar 75% varians residual. Lantai skor dari fitur sekitar 0,21.
5. EUI menyumbang sekitar 65% skor (NRMSE sekitar 0,65); prioritas utama.
6. Sel grid iklim dan urutan spasial sub_n memberi sinyal residual. TE sel iklim dan window, plus koreksi residual grup/kNN.
7. 77 baris test duplikat train dengan target identik (dup_fix).
8. Adopsi: N rumah tangga bisa direkonstruksi eksak dari hvac_dist; baris di subzone tak terlihat dimodelkan secara binomial.
9. Tidak ada drift train-test (AUC 0,50), jadi adversarial reweighting tidak perlu.
10. id tidak informatif dan tidak dipakai.
