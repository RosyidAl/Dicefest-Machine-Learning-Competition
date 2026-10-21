# Outline Presentasi Final

Format: 10 menit paparan + 10 menit tanya jawab (Zoom, 30 Oktober). Sepuluh slide. Bobot penilaian final: kebijakan
berbasis interpretabilitas 35, fitur domain dan pra-proses 25, slide dan tanya jawab 25, arsitektur dan kode 15. Porsi
waktu mengikuti bobot itu: sekitar 4 menit kebijakan, 2,5 menit fitur, 1,5 menit arsitektur, sisanya pembuka dan
penutup. Semua figur ada di `reports/figures/`.

## Alur slide

### Slide 1. Judul dan pesan utama (30 detik)
- "Dari Struktur Data ke Peta Kebijakan: Memprediksi Trilema Transisi Energi Perkotaan". Tim BISMILLAH.
- Satu kalimat: kami memprediksi lima indikator energi untuk 613 ribu kawasan (LB publik 0,24913), lalu
  menerjemahkan model menjadi peta prioritas per zona: di mana retrofit dulu, di mana heat pump sekarang, dan di mana
  jaringan listrik harus diperkuat lebih dulu.

### Slide 2. Masalah: trilema dalam lima angka (45 detik)
- Emisi (gas, EUI), keandalan jaringan (listrik), kualitas stok (rasio bangunan bersih), transisi teknologi (adopsi).
- Metrik rata-rata RMSE/sigma; EUI menyumbang separuh galat, jadi EUI adalah medan utama.
- Figur: `fig1_targets.png`.

### Slide 3. Data dulu, model kemudian (1 menit)
- Tiga identitas yang kami temukan sebelum melatih model apa pun: gas = EUI x luas lantai; rasio bersih dihitung eksak;
  adopsi konstan per subzona (92% baris test tinggal disalin; sisanya dari rekonstruksi jumlah rumah tangga).
- Split test acak per baris (AUC adversarial 0,50), jadi informasi wilayah sah dipakai asalkan out-of-fold.
- Pesan: satu target selesai tanpa model, gas dan EUI berbagi satu keluarga model.

### Slide 4. Fitur domain (1,5 menit) [25 poin]
- Rasio fisik: luas per meter (ukuran hunian), luas per sertifikat, pangsa pemanas sisa.
- Target sebagai intensitas (per meter, per luas, per sertifikat, log, Poisson) dengan bobot sesuai struktur noise.
- Encoding spasial out-of-fold: subzona, zona, sel grid iklim, jendela sepanjang nomor subzona.
- kNN-target di ruang stok bangunan: rata-rata target kawasan paling mirip di sekitarnya (perbaikan model tunggal
  terbesar: EUI -0,0022).
- Figur: `fig5_gains.png` (panel atas).

### Slide 5. Berapa yang bisa dijelaskan? (45 detik)
- Varians residual mengikuti a + b/M: tiga perempat galat adalah rata-rata rumah tangga yang tidak teramati.
- Implikasi: yang membantu adalah mengurangi varians (ensemble, bobot sesuai noise), bukan model yang lebih besar.
- Figur: `fig2_noise_floor.png`.

### Slide 6. Arsitektur multi-output (1,5 menit) [15 poin]
- Diagram `fig_arch.png`: tiga jalur (keluarga gas/EUI, keluarga listrik, identitas), 41 anggota (XGBoost, LightGBM,
  MLP/PLR/TabM, multi-task, retrieval NCA), blend dengan bobot yang bervariasi menurut luas, meter, dan level prediksi,
  koreksi residual spasial, pasca-proses (meter kecil, batas truncation gas).
- Validasi: setiap komponen pasca-blend di-fit di holdout internal yang terbukti memprediksi LB; notebook mereproduksi
  file submisi identik byte (sha256 aec745ab).

### Slide 7. Apa yang mendorong konsumsi (1,5 menit) [35 poin]
- Figur: `pol_families.png` dan `pol_dependence.png`.
- Ukuran hunian: hunian kecil sekitar 65% lebih boros gas per m², hunian besar sekitar 75% lebih boros listrik per rumah.
- Stok lama menaikkan EUI hingga 7%; iklim menggeser EUI sekitar 15%.
- Adopsi ditentukan infrastruktur: pangsa gas jaringan tinggi menekan adopsi; sosio-demografi hanya 15%.

### Slide 8. Peta trilema per zona (1,5 menit) [35 poin]
- Figur: `pol_trilemma.png` + Tabel 4 writeup.
- Q1 (118 zona, 47% gas): retrofit dulu, lalu elektrifikasi. Q2 (47 zona): heat pump sekarang. Q3 (47 zona): perkuat
  gardu dulu. Q4: program standar.
- Angka kunci: potensi hemat retrofit 7,4 TWh (5,2%), separuhnya di Q1; elektrifikasi penuh menaikkan beban listrik
  zona rata-rata 95%, tertinggi 121% (REG_290).

### Slide 9. Rekomendasi (1 menit) [35 poin]
1. Urutkan program per kuadran, jangan seragamkan.
2. Q2 sebagai kemenangan cepat heat pump.
3. Operator jaringan masuk sejak awal di Q1 dan Q3 (tarif waktu-pemakaian, penyimpanan panas).
4. Insentif khusus elektrifikasi di kawasan berjaringan gas (pangsa > 80%).
5. Pisahkan target efisiensi (hunian kecil, stok lama) dan target emisi absolut (zona bervolume gas besar).

### Slide 10. Penutup (30 detik)
- Tiga pelajaran: struktur data sebelum model; validasi yang meniru test; interpretasi yang bisa dieksekusi.
- Batasan: asosiasi bukan kausalitas; nama kolom anonim; noise rumah tangga membatasi akurasi.

## Persiapan tanya jawab
- **Kenapa tidak GroupKFold per zona?** Test acak per baris; fold per zona membuat target encoding tampak merusak (EUI
  0,833 vs 0,692) padahal di test justru membantu.
- **Apakah ada kebocoran?** Semua fitur bertarget dihitung out-of-fold dan lulus uji acak label; bobot pasca-blend
  di-fit di holdout yang tidak pernah melihat label baris yang dinilai.
- **Kenapa model penjelas terpisah, bukan SHAP ensemble?** Ensemble memakai encoding target yang bukan tuas kebijakan;
  model penjelas hanya memakai karakteristik terukur, dengan kesesuaian R² 0,49 sampai 0,99.
- **Seberapa yakin dengan skenario beban listrik?** Perhitungan fisik sederhana (80% gas untuk pemanasan, SCOP 3) di
  atas prediksi model; urutan antar zona lebih kuat daripada angka absolutnya.
- **Apa yang tidak berhasil?** Stacking level-2, blend terpisah subzona baru, koreksi nonlinear, 20 fold; semuanya
  tercatat di registry eksperimen.
