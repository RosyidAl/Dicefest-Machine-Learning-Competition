# Naskah 02: Rekomendasi Kebijakan dari Temuan SHAP

Sumber: notebook `final/deliverables/DICEFEST2026_BISMILLAH.ipynb`, bagian 7, dan modul `explain.py` yang tertanam di
notebook. Nomor sel dihitung dari 0, dengan nomor eksekusi Jupyter di dalam kurung:
- sel 41 (In [30]): model penjelas dan R²;
- sel 43 (In [31]): porsi kontribusi per kelompok fitur;
- sel 45 (In [32]): fitur teratas;
- sel 47 (In [33]): grafik dependensi;
- sel 49 dan 50 (In [34], In [35]): peta trilema dan tabel zona.

## A. Hasil verifikasi

| Klaim di deskripsi tugas | Status | Bukti di notebook |
|---|---|---|
| Model penjelas LightGBM dengan TreeSHAP | Sesuai | `explain.fit_explain`: satu LightGBM per target (800 putaran, `num_leaves` 127); SHAP dari `model.predict(..., pred_contrib=True)`, yaitu TreeSHAP eksak bawaan LightGBM |
| Sampel 20.000 baris | Sesuai, dengan catatan | `explain.shap_values(n=20000, seed=0)` dipakai untuk tabel fitur teratas (sel 45) dan grafik dependensi (sel 47). Peta zona (sel 49) memakai SHAP model EUI pada **seluruh** 613.290 baris latih + uji (`family_contrib`), bukan sampel |
| `area_per_meter` dominan untuk log EUI | Sesuai | peringkat 1, rata-rata \|SHAP\| 0,0895 (sel 45) |
| `area_per_meter` dominan untuk "log listrik" | Sesuai, nama perlu dikoreksi | Targetnya **log listrik per meter**, yaitu log(listrik total / `bld_dim_12`), bukan log listrik total. Peringkat 1, 0,0895 (sel 45) |
| `hvac_rest`, `hvac_dist_02`, `hvac_dist_09` dominan untuk adopsi | Sesuai | peringkat 1 sampai 3: 3,5653; 0,9053; 0,2691 poin persentase (sel 45) |

Kecocokan model penjelas pada data yang tidak dilihat (R², sel 41): log EUI 0,563; log gas per meter 0,747; log listrik
per meter 0,491; adopsi 0,991. Angka adopsi tinggi karena adopsi hampir tertentu oleh komposisi pemanas (identitas
akuntansi), bukan karena perilakunya terjelaskan.

## B. Arti variabel `hvac_*` menurut kode

Label berasal dari `explain.HVAC_NAMES`. Kode menyebutnya sebagai porsi jenis pemanas yang direkonstruksi dari
identitas jumlah rumah tangga. Keluaran sel 45 menuliskannya di kolom "makna (dugaan)", karena nama kolom asli
disamarkan panitia.

| Variabel | Arti menurut kode | Peringkat SHAP adopsi (sel 45) |
|---|---|---|
| `hvac_dist_01` | tanpa pemanas sentral | 4 (0,0968) |
| `hvac_dist_02` | gas jaringan (sentral) | 2 (0,9053) |
| `hvac_dist_03` | tangki/LPG | 8 (0,0697) |
| `hvac_dist_04` | minyak | 7 (0,0816) |
| `hvac_dist_05` | kayu | tidak masuk 10 besar |
| `hvac_dist_06` | bahan bakar padat | tidak masuk 10 besar |
| `hvac_dist_07` | energi terbarukan | tidak masuk 10 besar |
| `hvac_dist_08` | jaringan panas distrik | 5 (0,0953) |
| `hvac_dist_09` | dua jenis atau lebih (non-terbarukan) | 3 (0,2691) |
| `hvac_dist_10` | dua jenis atau lebih (terbarukan) | tidak masuk 10 besar |
| `hvac_rest` | sisa porsi (listrik dan lainnya) = 100 − jumlah `hvac_dist_01` sampai `hvac_dist_10` (`common.base_features`) | 1 (3,5653) |

Konsekuensinya, adopsi pemanas listrik adalah bagian dari `hvac_rest`, dan semua porsi berjumlah 100. Karena itu, SHAP
komposisi pemanas sebagian besar mencerminkan pembagian porsi, bukan pendorong keputusan. Korelasi `hvac_rest` dengan
adopsi 0,988 (sel 29).

## C. Temuan SHAP yang dipakai

Rentang kontribusi dibaca dari garis rata-rata (merah) pada grafik sel 47, jadi angkanya perkiraan. Skala SHAP untuk
EUI dan listrik adalah logaritma, sedangkan untuk adopsi berupa poin persentase.

| Variabel | Target | Rata-rata \|SHAP\| (sel 45) | Rentang kontribusi (sel 47) | Bacaan |
|---|---|---|---|---|
| `area_per_meter` | log EUI | 0,0895 | sekitar +0,27 (rumah ±27 m² per meter) sampai −0,24 (±300 m²) | rumah kecil memakai gas per m² sekitar 65% lebih banyak (teks bagian 7.1) |
| `area_per_meter` | log listrik per meter | 0,0895 | sekitar −0,24 (±27 m²) sampai +0,31 (±300 m²) | rumah besar memakai listrik per rumah sekitar 75% lebih banyak (teks bagian 7.1) |
| `bld_era_15` | log EUI | 0,0198 | sekitar −0,01 (porsi 0%) sampai +0,07 (porsi ±90%) | bangunan periode ini lebih boros |
| `bld_era_09` | log EUI | 0,0153 | sekitar −0,01 sampai +0,04 | pola sama, lebih lemah |
| `env_climate_05` | log EUI | 0,0321 | sekitar −0,09 (nilai ±36) sampai +0,06 (nilai ±56) | iklim menggeser EUI sekitar 15% |
| `hvac_dist_02` | adopsi | 0,9053 | sekitar +4,3 poin (porsi gas jaringan ±35%), mendekati 0 pada ±80%, sekitar −1,3 poin pada ±95% | adopsi saat ini berada di kawasan yang belum tersambung gas |

Porsi kontribusi per kelompok (sel 43, dibaca dari grafik):
- Adopsi: komposisi pemanas sekitar 81% dan sosiodemografi sekitar 15%. Angka 81% juga tertulis di teks bagian 7.
- Listrik per meter: sosiodemografi sekitar 40% dan rasio fisik sekitar 26%.

## D. Tiga rekomendasi kebijakan

Disusun dari sudut pandang konsultan tata kota dan transisi energi. Setiap rekomendasi menyasar **kawasan**, bukan
variabel. Angka zona berasal dari keluaran sel 49 dan 50.

### Rekomendasi 1. Dahulukan *retrofit* selubung di kawasan rumah kecil dan bangunan periode lama, mulai dari zona Q1
- **Temuan SHAP.** `area_per_meter` adalah pendorong terbesar EUI. Kontribusinya +0,27 pada rumah kecil dan −0,24 pada
  rumah besar. `bld_era_15` (sampai +0,07), `bld_era_09` (sampai +0,04), dan `env_climate_05` (sampai +0,06) menambah
  lapisan berikutnya.
- **Mengapa langsung terkait.** Potensi penghematan per zona di notebook dihitung dari kontribusi SHAP stok bangunan
  (kelompok era bangunan, dimensi bangunan, dan rasio fisik) yang berada di atas rata-rata (`explain.zone_indices`).
- **Kawasan sasaran.**
  - Total potensi 7,39 TWh atau 5,23% dari 141,35 TWh konsumsi gas.
  - Kuadran Q1 (118 zona) memuat 66,1 TWh gas (46,8%) dan 3,65 TWh potensi.
  - Kandidat awal: REG_273 (hemat 7,36%), REG_270 (8,60%), dan REG_045 (6,69%), yang berada di peringkat teratas
    potensi absolut, serta REG_240 (9,34%) dengan persentase tertinggi di tabel tersebut.
- **Tindakan.**
  1. Program perbaikan dinding, atap, dan jendela, disertai audit energi gratis bagi kawasan dengan rumah kecil dan
     porsi bangunan periode lama yang tinggi di zona-zona tersebut.
  2. Di Q1, *retrofit* dijalankan sebelum subsidi pompa panas. Kebutuhan panas yang turun lebih dulu memperkecil
     tambahan listrik saat elektrifikasi.
- **Ukuran keberhasilan.** Penurunan konsumsi gas per m² di kawasan peserta, dibandingkan kawasan serupa yang belum
  ikut program.

### Rekomendasi 2. Pastikan kesiapan jaringan sebelum subsidi pompa panas di zona dengan lonjakan konsumsi listrik tertinggi
- **Temuan SHAP.** Untuk listrik per meter, kontribusi `area_per_meter` naik dari sekitar −0,24 menjadi +0,31. Artinya,
  rumah besar memakai listrik per rumah sekitar 75% lebih banyak. Kawasan rumah besar sudah memiliki konsumsi per
  sambungan tertinggi sebelum ada elektrifikasi.
- **Kawasan sasaran.**
  - Skenario di notebook mengganti pemanas gas dengan pompa panas (80% gas untuk pemanasan, COP 3). Hasilnya, konsumsi
    listrik tahunan zona naik dengan median 94,85%.
  - Tertinggi: REG_290 +121,41%, REG_259 +115,10%, REG_275 +114,29%, REG_260 +113,91%, REG_270 +113,46%. Semuanya Q1.
  - Q1 rata-rata +102,5% dan Q3 +99,1%.
  - Di dalam zona-zona ini, kawasan dengan rumah besar didahulukan dalam penilaian gardu.
- **Tindakan.**
  1. Operator jaringan menilai kapasitas gardu di zona sasaran lebih dulu.
  2. Penguatan jaringan dijadwalkan.
  3. Subsidi pompa panas dibuka bertahap, disertai tarif berbasis waktu pemakaian dan pompa panas dengan penyimpan
     panas.
- **Ukuran keberhasilan.** Tidak ada gardu yang melampaui kapasitas setelah tahap subsidi dibuka; ini diukur oleh
  operator jaringan.
- **Batas klaim.**
  - Angka kenaikan adalah energi tahunan, bukan beban puncak.
  - Label "kenaikan beban" pada grafik dan tabel notebook berarti kenaikan konsumsi tahunan.
  - Label Q2 "jaringan masih longgar" pada tabel keluaran adalah label lama; data tidak memuat kapasitas jaringan.

### Rekomendasi 3. Program elektrifikasi khusus untuk kawasan yang sudah tersambung jaringan gas
- **Temuan SHAP.**
  - Tiga variabel teratas untuk adopsi adalah `hvac_rest` (3,5653), `hvac_dist_02` (0,9053), dan `hvac_dist_09`
    (0,2691).
  - Kontribusi porsi gas jaringan sekitar +4,3 poin pada porsi ±35%, lalu negatif di atas ±80%.
  - Sosiodemografi hanya sekitar 15% dari kontribusi.
- **Cara membaca yang jujur.**
  - Komposisi pemanas adalah identitas akuntansi: makin besar porsi gas jaringan, makin kecil sisa porsi untuk listrik.
  - Karena itu, rekomendasi ini **tidak** menyatakan bahwa mengubah porsi pemanas akan menaikkan adopsi.
  - Yang dipakai adalah polanya: adopsi saat ini hampir hanya ada di kawasan yang belum tersambung gas, dan
    karakteristik warga tidak banyak menjelaskan perbedaannya.
- **Kawasan sasaran.**
  - Kawasan dengan porsi pemanas gas jaringan di atas 80%.
  - Contoh dari tabel sel 50: REG_249 (85,54%), REG_330 (85,21%, adopsi 2,39%), REG_260 (84,80%), dan REG_045 (83,58%,
    adopsi 2,60%).
- **Tindakan.**
  1. Paket insentif berupa bantuan biaya pemasangan pompa panas dan pembongkaran ketel lama.
  2. Dipadukan dengan Rekomendasi 1 supaya pompa panas bekerja di rumah yang sudah diperbaiki.
  3. Dijalankan sebagai pilot bertahap: zona dibuka bergiliran supaya dampaknya bisa dibandingkan dengan zona yang
     belum ikut. Dengan begitu, penyebab rendahnya adopsi bisa diuji, bukan hanya diduga.
- **Ukuran keberhasilan.** Kenaikan adopsi per subzona di kawasan peserta dibandingkan kawasan pembanding.

## E. Naskah lisan (sekitar 520 kata, 4 menit)

Bagian ini menjawab pertanyaan yang paling penting bagi pemerintah kota: setelah model memprediksi, apa yang harus
dilakukan? Untuk itu kami melatih satu model penjelas per target. Model ini hanya memakai karakteristik kawasan yang
bisa diukur, lalu kami baca dengan SHAP. SHAP adalah metode yang membagi setiap prediksi menjadi sumbangan dari
masing-masing variabel.

Supaya temuan itu bisa langsung dipakai, kami menggabungkan nilai aktual kawasan latih dan prediksi kawasan uji, lalu
meringkasnya menjadi 330 zona. Setiap zona dipetakan menurut dua ukuran: intensitas gasnya, dan kenaikan konsumsi
listriknya bila pemanas gas diganti pompa panas. Garis median membagi zona menjadi empat kuadran, dan setiap kuadran
membutuhkan urutan program yang berbeda.

Temuan pertama: ukuran rumah adalah pendorong terbesar. Untuk intensitas gas, sumbangan area per meter turun dari
sekitar plus 0,27 pada rumah seluas 27 meter persegi menjadi minus 0,24 pada rumah seluas 300 meter persegi, dalam
skala logaritma. Artinya rumah kecil memakai gas per meter persegi sekitar 65 persen lebih banyak. Periode bangunan dan
iklim menambah lapisan berikutnya.

Dari sini lahir rekomendasi pertama: dahulukan perbaikan selubung bangunan, yaitu dinding, atap, dan jendela, di
kawasan dengan rumah kecil dan bangunan periode lama. Potensinya kami hitung per zona dari sumbangan SHAP stok
bangunan: 7,39 terawatt-jam, atau 5,23 persen konsumsi gas. Hampir separuhnya, 3,65 terawatt-jam, ada di 118 zona
kuadran pertama. Kandidat awalnya REG_273, REG_270, dan REG_045.

Temuan kedua: untuk listrik, arahnya terbalik. Sumbangan area per meter naik dari sekitar minus 0,24 menjadi plus
0,31, jadi rumah besar memakai listrik per rumah sekitar 75 persen lebih banyak. Rekomendasi kedua: siapkan jaringan
sebelum subsidi pompa panas. Bila pemanas gas diganti pompa panas, konsumsi listrik tahunan zona naik dengan median
94,85 persen, dan tertinggi 121 persen di REG_290. Di zona seperti ini, operator jaringan perlu menilai kapasitas gardu
lebih dulu, terutama di kawasan rumah besar yang konsumsi per sambungannya sudah paling tinggi. Setelah itu subsidi
dibuka bertahap, disertai tarif berbasis waktu pemakaian.

Temuan ketiga menyangkut adopsi pemanas listrik. Variabel teratasnya adalah sisa porsi pemanas dan porsi pemanas gas
jaringan. Di sini kami harus jujur: ini sebagian besar identitas akuntansi. Makin besar porsi gas jaringan, makin kecil
sisa porsi untuk listrik. Jadi kami tidak mengatakan bahwa mengubah variabel itu akan menaikkan adopsi. Yang bisa
dibaca adalah polanya. Sumbangan porsi gas jaringan sekitar plus 4 poin pada porsi 35 persen, lalu negatif di atas 80
persen, sedangkan sosiodemografi hanya menyumbang sekitar 15 persen. Artinya, adopsi saat ini hampir hanya terjadi di
kawasan yang belum tersambung gas. Rekomendasi ketiga: buat program khusus untuk kawasan yang sudah tersambung jaringan
gas, misalnya bantuan biaya pemasangan pompa panas dan pembongkaran ketel lama, digabung dengan perbaikan bangunan.
Jalankan dulu sebagai pilot bertahap di beberapa zona, misalnya REG_045 yang adopsinya baru 2,6 persen, supaya
dampaknya bisa diukur.

Ringkasnya: perbaiki bangunan lebih dulu di kuadran pertama, siapkan jaringan sebelum elektrifikasi, dan beri dorongan
khusus di kawasan yang sudah tersambung gas.

Satu catatan keterbatasan. SHAP menunjukkan seberapa besar sumbangan setiap variabel terhadap prediksi model, bukan
hubungan sebab-akibat. Skenario listrik juga menghitung energi tahunan dengan asumsi 80 persen gas untuk pemanasan dan
efisiensi pompa panas 3, bukan beban puncak. Karena itu, urutan prioritas antarzona lebih layak dipegang daripada angka
absolutnya.

## F. Catatan sumber angka untuk tim

- **Angka yang tercetak di keluaran notebook:**
  - rata-rata \|SHAP\| (sel 45);
  - R² (sel 41);
  - tabel kuadran, total 141,35 TWh, 7,39 TWh, 5,23%, median 94,85%, dan tabel zona teratas (sel 49 dan 50).
- **Angka yang dibaca dari grafik,** sehingga sebaiknya diucapkan dengan kata "sekitar":
  - rentang kontribusi SHAP (sel 47);
  - porsi kelompok 81% dan 15% (sel 43).
- **Angka 65% dan 75%** berasal dari teks bagian 7.1 di notebook. Angka ini konsisten dengan grafik: selisih
  kontribusi 0,51 dan 0,55 dalam skala log setara dengan faktor 1,67 dan 1,73.
