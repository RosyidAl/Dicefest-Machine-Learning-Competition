# Menyerahkan Deck ke Claude Design

Claude Design adalah produk Anthropic Labs (pratinjau riset sejak April 2026) yang membuat slide, prototipe, dan desain
di atas kanvas dari prompt teks. Informasi di bawah sebagian berasal dari ulasan pihak ketiga, jadi cocokkan dengan
tampilan aplikasi yang sedang dipakai.

## Prinsip
- Claude Design paling baik bila diberi isi yang sudah jadi. Bila hanya diberi writeup panjang, ia akan meringkas
  sendiri dan bisa salah memilih angka atau pesan. Serahkan brief lengkap: prompt utama, isi per slide, gambar, dan
  daftar periksa angka.
- Sebutkan kode warna dan aturan huruf secara eksplisit. Menurut beberapa ulasan, Claude Design memakai selera desain
  umum dan tidak otomatis mengikuti identitas visual tertentu.
- Minta rancangan tata letak sebelum slide dibuat, lalu revisi per slide dengan perintah spesifik.

## Langkah
1. Buka proyek baru jenis presentasi atau slide deck.
2. Tempel prompt utama dari brief, lalu unggah isi per slide dan semua gambar dalam satu pesan.
3. Minta: "Usulkan dulu tata letak setiap slide secara singkat, belum perlu dibuat." Periksa urutan dan pesan.
4. Minta pembuatan deck. Setelah jadi, revisi per slide, misalnya:
   - "Slide 8: perbesar gambar dependensi hingga dua pertiga slide, sisakan tiga butir, tampilkan 65% dan 75% sebagai
     angka besar."
   - "Semua judul slide: pakai kalimat pernyataan dari brief, jangan diganti."
   - "Hapus ikon dekoratif yang tidak membawa informasi."
5. Minta pemeriksaan angka: "Cocokkan setiap angka di semua slide dengan daftar periksa angka; laporkan yang berbeda
   atau tidak ada di daftar."
6. Ekspor di akhir saja. Untuk presentasi daring, PDF paling aman. PPTX atau penyerahan ke Canva hanya bila deck masih
   perlu disunting di alat lain.

## Hal yang perlu diwaspadai
- Ekspor PPTX dilaporkan sering mengubah huruf, posisi teks, atau meratakan teks menjadi gambar. Sisihkan waktu untuk
  merapikan bila memakai PPTX.
- Ekspor langsung ke Google Slides dilaporkan tidak tersedia.
- Claude Design bisa membulatkan atau menulis ulang angka saat merapikan teks; karena itu langkah 5 wajib.
- Gambar beresolusi rendah akan terlihat buram bila diperbesar. Render ulang gambar penting dengan lebar minimal sekitar
  1.600 piksel dan huruf sumbu yang lebih besar sebelum diunggah.
- Belum ada konfirmasi resmi bahwa Claude Design memuat skill kustom. Karena itu brief harus berdiri sendiri: semua
  aturan penting dari skill ini ikut ditulis di prompt utama.
