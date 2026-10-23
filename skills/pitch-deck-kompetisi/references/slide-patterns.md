# Pola Slide untuk Deck Final Kompetisi ML

Setiap pola berisi kegunaan, tata letak, dan kesalahan yang sering terjadi. Pilih satu pola per slide. Bila dua pola
terasa perlu, biasanya slide itu membawa dua pesan dan sebaiknya dipecah.

## Daftar isi
1. Judul
2. Masalah dan metrik
3. Wawasan data
4. Skema validasi
5. Fitur
6. Batas akurasi atau analisis galat
7. Arsitektur
8. Hasil
9. Interpretasi (SHAP)
10. Peta atau segmentasi kebijakan
11. Rekomendasi
12. Penutup
13. Slide cadangan

## 1. Judul
- **Tata letak:** judul-kalimat yang menyatakan inti kontribusi, nama tim, nama acara, satu atau dua angka hasil
  dalam ukuran besar.
- **Kesalahan umum:** judul yang hanya menyebut topik lomba; logo dan dekorasi yang memakan ruang.

## 2. Masalah dan metrik
- **Tata letak:** tabel kecil yang memetakan target ke tujuan nyata (misalnya emisi, keandalan jaringan, biaya),
  ditambah satu baris tentang metrik dan target yang paling menentukan skor.
- **Kesalahan umum:** menyalin deskripsi lomba yang sudah diketahui juri.

## 3. Wawasan data
- **Tata letak:** tiga kotak temuan berdampingan, masing-masing berupa satu kalimat dan satu angka bukti. Tanpa gambar
  bila temuannya berupa identitas atau rumus.
- **Kesalahan umum:** statistik deskriptif umum (rata-rata, histogram) yang tidak mengubah keputusan apa pun. Hanya
  tampilkan temuan yang mengubah desain solusi.

## 4. Skema validasi
- **Tata letak:** perbandingan dua skema dalam tabel kecil atau dua angka besar ("per zona 0,8331 vs acak 0,6699"),
  plus grafik kecocokan validasi dengan *leaderboard*.
- **Kesalahan umum:** menyebut "5-fold CV" tanpa menjelaskan mengapa skema itu meniru data uji.

## 5. Fitur
- **Tata letak:** empat butir kelompok fitur dengan alasan domain singkat, dan grafik kontribusi atau ablasi di sisi
  lain. Sorot satu fitur dengan perbaikan terbesar sebagai angka besar.
- **Kesalahan umum:** daftar nama kolom; klaim fitur "penting" tanpa ukuran perbaikan.

## 6. Batas akurasi atau analisis galat
- **Tata letak:** satu grafik (misalnya galat terhadap ukuran kelompok) dan satu kalimat implikasi.
- **Kesalahan umum:** grafik tanpa kesimpulan; juri tidak sempat menafsirkan sendiri.

## 7. Arsitektur
- **Tata letak:** diagram alur kiri ke kanan (data, fitur, model, penggabungan, koreksi, submisi) yang mengisi
  sebagian besar slide, ditambah tiga butir: jumlah dan jenis model, cara penggabungan, bukti reprodusibilitas.
- **Kesalahan umum:** tabel hiperparameter; nama varian internal (misalnya "v19vc", "xgb_gpn_tew") yang tidak bermakna
  bagi juri. Rincian masuk slide cadangan.

## 8. Hasil
- **Tata letak:** skor dan posisi sebagai angka besar, grafik perkembangan skor, dan satu kalimat tentang konsistensi
  validasi dengan *leaderboard*.
- **Kesalahan umum:** hanya skor publik; tidak menyebut stabilitas atau perbedaan publik dan private.

## 9. Interpretasi (SHAP)
- **Tata letak:** satu grafik dependensi atau porsi kontribusi yang besar, dengan dua atau tiga angka efek yang mudah
  diucapkan ("rumah kecil +65% gas per m²"). Keterangan kecil: model penjelas, kecocokannya, dan bahwa ini asosiasi.
- **Kesalahan umum:** grafik ringkasan SHAP penuh dengan puluhan fitur anonim; klaim sebab-akibat.

## 10. Peta atau segmentasi kebijakan
- **Tata letak:** peta atau diagram kuadran di satu sisi, tabel ringkas kuadran di sisi lain (jumlah wilayah, porsi,
  potensi, langkah). Angka total sebagai angka besar di atas.
- **Kesalahan umum:** label kuadran yang menyiratkan data yang tidak dimiliki (misalnya kapasitas jaringan); asumsi
  skenario tidak disebut.

## 11. Rekomendasi
- **Tata letak:** tiga sampai lima rekomendasi bernomor, masing-masing diawali kata kerja, menyebut sasaran (wilayah,
  kelompok) dan urutan. Baris keterbatasan kecil di bawah.
- **Kesalahan umum:** rekomendasi umum yang bisa ditulis tanpa model ("tingkatkan efisiensi energi").

## 12. Penutup
- **Tata letak:** tiga pelajaran atau kontribusi dalam satu baris masing-masing, kontak tim, dan ajakan tanya jawab.
- **Kesalahan umum:** slide "Terima kasih" kosong yang menghabiskan waktu.

## 13. Slide cadangan
Siapkan untuk pertanyaan yang hampir pasti muncul: rincian skema validasi, komposisi model, sensitivitas asumsi
skenario, dan model untuk target khusus. Slide cadangan boleh lebih padat karena hanya dibuka saat ditanya.
