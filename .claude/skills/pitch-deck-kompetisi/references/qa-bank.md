# Bank Tanya Jawab Juri untuk Kompetisi ML

Pertanyaan berikut paling sering muncul di babak final kompetisi data science. Untuk setiap deck, pilih 8 sampai 12
yang relevan, tulis jawaban 2 sampai 3 kalimat dengan angka dari lembar fakta, dan tandai slide cadangan rujukannya.
Jawaban yang baik dimulai dengan inti jawaban, lalu bukti, lalu batasannya.

## Validasi dan kebocoran
1. **Bagaimana memastikan tidak ada kebocoran data?** Sebut fitur mana saja yang memakai target, bahwa semuanya
   dihitung secara *out-of-fold*, dan uji yang dipakai untuk memeriksanya (misalnya uji label acak).
2. **Mengapa skema validasi ini, bukan per wilayah atau per waktu?** Jelaskan mekanisme pemisahan data uji yang
   ditemukan dan bukti pendukungnya (porsi kelompok baru, validasi adversarial).
3. **Apakah model terlalu menyesuaikan diri dengan *leaderboard* publik?** Tunjukkan bahwa keputusan diambil dari
   *holdout set* internal dan bahwa perkiraannya konsisten dengan perubahan skor; sebut skor private bila sudah ada.

## Model dan arsitektur
4. **Mengapa banyak model, bukan satu model yang dioptimalkan?** Kaitkan dengan struktur galat (misalnya galat acak
   yang besar) dan besar perbaikan dari penggabungan.
5. **Apa yang dicoba tetapi tidak berhasil?** Siapkan dua atau tiga contoh dengan alasan singkat; ini menunjukkan
   proses yang jujur.
6. **Apakah hasilnya bisa direproduksi?** Sebut seed, mode deterministik, dan hash berkas submisi yang identik.
7. **Berapa biaya komputasinya?** Sebut perangkat keras dan durasi pelatihan, serta jalur cepat tanpa pelatihan bila
   ada.

## Interpretasi dan kebijakan
8. **Apakah SHAP menunjukkan sebab-akibat?** Tidak; SHAP menunjukkan asosiasi pada model. Jelaskan cara membacanya dan
   apa yang dibutuhkan untuk klaim kausal (misalnya data program atau eksperimen).
9. **Mengapa memakai model penjelas terpisah, bukan SHAP dari *ensemble*?** Fitur *ensemble* memuat statistik target
   wilayah yang bukan tuas kebijakan; model penjelas hanya memakai karakteristik yang bisa diintervensi, dengan
   kecocokan yang dilaporkan.
10. **Seberapa yakin dengan angka skenario?** Sebut asumsi, rentang sensitivitas, dan bagian yang tidak berubah
    (misalnya peringkat wilayah).
11. **Bagaimana rekomendasi ini dijalankan di lapangan?** Sebut sasaran, urutan, pihak yang terlibat, dan data tambahan
    yang dibutuhkan sebelum eksekusi.

## Data
12. **Bagaimana menangani kolom yang namanya disamarkan?** Jelaskan cara menduga maknanya dari pola data dan bahwa
    tafsiran itu dinyatakan sebagai dugaan.
13. **Apa keterbatasan terbesar data ini?** Sebut satu atau dua (misalnya variasi antarrumah tangga yang tidak
    teramati, tidak adanya data beban puncak) beserta dampaknya pada kesimpulan.
