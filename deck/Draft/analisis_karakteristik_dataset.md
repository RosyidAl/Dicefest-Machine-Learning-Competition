# Analisis Karakteristik Dataset

Sebelum membuat model, tim memeriksa datanya terlebih dahulu, dan dari pemeriksaan itu kita mendapatkan cukup banyak hal.

Pertama, kita jadi tahu apa yang sebenarnya diwakili satu baris data. Satu baris bukan satu rumah, melainkan satu kawasan yang berisi rata-rata 22 meteran, kira-kira setara 22 rumah tangga. Dari sini terlihat bahwa konsumsi gas dan listrik sebuah kawasan sebagian besar ditentukan oleh besar kecilnya kawasan itu. Kawasan dengan banyak rumah pasti memakai energi lebih banyak. Artinya, kalau ingin memahami perilaku konsumsi, kita perlu melihat konsumsi per rumah tangga atau per meter persegi, bukan total kawasannya saja.

Kedua, meskipun semua nama kolom disamarkan panitia, kita berhasil menemukan arti beberapa kolom penting hanya dari pola angkanya. Kolom luas lantai ditemukan karena gas selalu sama persis dengan EUI dikali kolom tersebut. Kolom jumlah meteran, jumlah sertifikat energi, dan koordinat iklim juga ditemukan dengan cara serupa. Hanya status kota atau desa yang artinya langsung terbaca, karena kolomnya sudah berisi teks "Urban" dan "Rural". Tanpa temuan ini, kita tidak bisa membuat fitur yang bermakna secara fisik.

Ketiga, kita menemukan bahwa tidak semua target perlu ditebak dengan model. Porsi bangunan bersih ternyata selalu bisa dihitung dengan rumus dari dua kolom yang ada. Adopsi pemanas listrik selalu bernilai sama dalam satu subzona, sehingga untuk sebagian besar kawasan uji jawabannya cukup disalin. Gas dan EUI ternyata satu besaran yang ditulis dalam dua bentuk, sehingga keduanya cukup dipelajari oleh satu kelompok model yang sama; hanya cara menggabungkan prediksinya yang disesuaikan untuk masing-masing. Jadi dari lima target, yang benar-benar perlu dipelajari model hanya tiga, dan tenaga bisa dipusatkan di sana.

Keempat, kita jadi tahu bagaimana panitia membagi data latih dan data uji, yaitu secara acak per baris. Akibatnya, hampir semua kawasan uji punya tetangga di subzona yang sama dalam data latih. Kita juga menemukan bahwa nomor subzona berurutan sesuai letaknya, sehingga kawasan yang berdekatan bisa dicari walaupun tidak ada koordinat. Dua temuan ini menentukan cara menguji model dengan benar, sekaligus membuka jalan untuk memakai informasi dari kawasan sekitar.

Kelima, kita tahu bahwa datanya bersih. Nilai kosong sangat sedikit, hanya ada beberapa kolom yang tidak berguna, dan ada 77 kawasan uji yang kembar dengan kawasan latih. Pembersihan yang diperlukan pun sedikit.

Terakhir, dari pemeriksaan galat kita tahu batas kemampuan data ini. Sebagian besar kesalahan berasal dari perbedaan kebiasaan antarrumah tangga yang tidak tercatat di data, sehingga ada batas skor sekitar 0,21 yang tidak bisa dilewati model apa pun.

Jadi, hasil utama dari menganalisis karakteristik dataset adalah arah kerja yang jelas: target mana yang perlu model, fitur apa yang masuk akal dibuat, bagaimana cara menguji model, dan sampai mana hasilnya bisa diperbaiki.
