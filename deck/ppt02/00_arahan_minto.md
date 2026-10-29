# Arahan pengguna (27 Oktober 2026): perombakan pitch deck top-down berbasis Prinsip Piramida Minto

Salinan apa adanya dari arahan yang menjadi dasar `01_struktur_deck_minto.md`. Disimpan untuk keterlacakan; teks di
bawah garis tidak disunting.

---

# PROMPT FINAL — PEROMBAKAN PITCH DECK BERBASIS TOP-DOWN DAN PRINSIP PIRAMIDA MINTO

## 1. Peran dan Tujuan

Bertindaklah sebagai **konsultan strategi presentasi tingkat eksekutif, ahli komunikasi kebijakan publik, dan reviewer teknis machine learning**. Tugas Anda adalah merombak struktur *pitch deck* yang tersedia menjadi presentasi yang persuasif, berbasis bukti, mudah dipahami oleh audiens campuran, dan siap dibawakan dalam kompetisi atau forum pengambilan keputusan.

Gunakan pendekatan **Top-Down versi ringan berdasarkan Prinsip Piramida Minto**: sampaikan kesimpulan dan rekomendasi terlebih dahulu, kemudian uraikan bukti pendukung, metodologi, implikasi kebijakan, serta keterbatasan analisis.

Audiens terdiri atas dua kelompok utama:

- **Juri pembuat kebijakan:** berfokus pada urgensi masalah, relevansi rekomendasi, kelayakan implementasi, dan dampak publik.
- **Juri teknis:** berfokus pada validitas rekayasa fitur, arsitektur model, konsistensi angka, dan kehati-hatian dalam menafsirkan hasil analisis.

Tujuan utama presentasi adalah membuat kedua kelompok memahami bahwa rekomendasi kebijakan yang diajukan memiliki dasar analitis yang memadai, tanpa membebani audiens dengan detail teknis yang tidak diperlukan dalam presentasi utama.

## 2. Batas Waktu dan Alokasi Durasi

Susun presentasi utama dengan durasi bicara **tepat 9 menit 30 detik**, dalam slot presentasi total 10 menit. Sisakan 30 detik sebagai *buffer* di luar narasi utama untuk transisi, jeda, atau gangguan teknis.

Gunakan pembagian waktu berikut tanpa mengubah urutannya:

| Bagian                             | Rentang waktu |           Durasi | Bobot/fokus                               |
| ---------------------------------- | ------------- | ---------------: | ----------------------------------------- |
| 1. Pembukaan dan Executive Summary | 00:00–01:00   |          1 menit | Trilema Energi dan tiga rekomendasi utama |
| 2. Rekayasa Fitur                  | 01:00–03:30   | 2 menit 30 detik | Bobot 25                                  |
| 3. Arsitektur Model                | 03:30–05:00   | 1 menit 30 detik | Bobot 15                                  |
| 4. Rekomendasi Kebijakan Publik    | 05:00–08:30   | 3 menit 30 detik | Bobot 35                                  |
| 5. Keterbatasan dan Penutup        | 08:30–09:30   |          1 menit | Transparansi dan kesimpulan               |

Alokasi waktu di atas merupakan batas yang harus dipatuhi. Tetapkan durasi untuk setiap slide, pastikan jumlahnya tepat 9 menit 30 detik, dan sesuaikan panjang narasi dengan kecepatan bicara presentasi yang wajar, sekitar 120–140 kata per menit. Untuk bagian pembukaan, buat naskah yang dirancang khusus agar dapat disampaikan dalam 60 detik.

**Jangan menghitung slide cadangan Q&A sebagai bagian dari durasi presentasi utama.**

## 3. Instruksi Struktur Slide-by-Slide

### BAGIAN I — PEMBUKAAN DAN EXECUTIVE SUMMARY

**00:00–01:00**

Mulailah dengan masalah strategis, bukan penjelasan model.

Buka dengan **Trilema Energi** dan jelaskan secara ringkas mengapa permasalahan tersebut membutuhkan perhatian serta keputusan kebijakan. Setelah itu, langsung ungkapkan tiga rekomendasi kebijakan publik utama yang dihasilkan dari analisis.

Ketentuan:

- Tiga rekomendasi harus muncul pada menit pertama.
- Gunakan rekomendasi yang benar-benar terdapat dalam materi sumber. Jangan menciptakan rekomendasi atau klaim baru.
- Nyatakan tesis utama presentasi dalam satu kalimat yang kuat dan mudah diingat.
- Jangan membahas algoritma, proses rekayasa fitur, atau arsitektur model.
- Susun narasi pembukaan agar dapat dihafalkan dan disampaikan dengan percaya diri.
- Buat transisi yang jelas dari masalah energi menuju rekomendasi, tanpa membocorkan seluruh rincian teknis.

Target hasil: audiens memahami masalah yang dihadapi, arah solusi yang ditawarkan, dan alasan untuk mendengarkan penjelasan berikutnya.

### BAGIAN II — REKAYASA FITUR

**01:00–03:30 | Bobot 25**

Tunjukkan bahwa rekomendasi yang telah diperkenalkan bukan sekadar opini, tetapi didukung oleh proses pengolahan data dan rekayasa fitur yang dirancang untuk meningkatkan keandalan analisis.

Fokuskan pembahasan pada dua keunggulan berikut.

**A. Fitur fisik: `area_per_meter`**

- Jelaskan definisi serta peran fitur secara sederhana.
- Tekankan bagaimana pencegahan *data leakage* diterapkan pada fitur ini.
- Jelaskan mengapa integritas fitur penting bagi validitas prediksi.
- Gunakan ilustrasi visual yang konkret, bukan uraian rumus yang panjang.

**B. Fitur spasial: kNN out-of-fold berlapis**

- Jelaskan bagaimana informasi kedekatan spasial dimanfaatkan.
- Uraikan secara singkat penggunaan pendekatan *out-of-fold* untuk mengurangi risiko kebocoran informasi target.
- Terangkan alasan penggunaan pendekatan berlapis, sesuai implementasi yang benar-benar dilakukan.
- Hubungkan kualitas fitur dengan kemampuan analisis untuk mendukung pemetaan prioritas kebijakan.

Gunakan pola komunikasi berikut: **fitur yang dibangun → risiko yang dikendalikan → kontribusi terhadap kualitas analisis**.

Hindari klaim bahwa teknik tersebut otomatis menjamin model bebas dari seluruh bentuk kebocoran atau bias. Gunakan hanya penjelasan yang didukung oleh implementasi dan dokumentasi proyek.

### BAGIAN III — ARSITEKTUR MODEL

**03:30–05:00 | Bobot 15**

Jelaskan arsitektur model melalui diagram tiga jalur yang sederhana. Utamakan kejelasan alur dan fungsi setiap jalur dibandingkan rincian algoritma.

**Wajib menjaga konsistensi jumlah model sebagai berikut:**

1. **Jalur Gas/EUI:** 28 model yang masuk ke proses *blending*.
2. **Jalur Listrik:** 13 model yang masuk ke proses *blending*.
3. **Jalur adopsi di subzona baru:** 2 model LightGBM terpisah yang secara khusus memprediksi adopsi pada 8,14% subzona baru.

Sampaikan angka-angka tersebut secara eksplisit dan konsisten:

- Total model dalam keseluruhan arsitektur adalah **43 model**.
- Sebanyak **41 model** digabungkan dalam proses *blending* berbobot adaptif.
- Dua model LightGBM lainnya merupakan jalur terpisah dan **tidak termasuk dalam 41 model yang di-blend bersama**.

Tampilkan hubungan antarkomponen menggunakan diagram alur yang mudah dipahami, idealnya dengan tiga jalur yang dibedakan secara visual.

Jangan menyederhanakan diagram sampai mengaburkan pemisahan antara 41 model yang di-*blend* dan 2 model LightGBM yang terpisah. Jangan pula menyatakan bahwa seluruh 43 model berada dalam satu ensemble gabungan.

Gunakan bahasa yang dapat dipahami audiens nonteknis, tetapi tetap cukup presisi untuk dipertanggungjawabkan di hadapan juri teknis.

### BAGIAN IV — REKOMENDASI KEBIJAKAN PUBLIK

**05:00–08:30 | Bobot 35**

Jadikan bagian ini sebagai inti argumentasi presentasi. Pindahkan fokus dari cara model dibangun ke bagaimana hasil analisis dapat digunakan untuk membantu pengambilan keputusan.

Bahas empat komponen berikut berdasarkan materi dan hasil analisis yang tersedia.

**A. Peta 330 zona**

- Perlihatkan peta yang membantu audiens memahami distribusi kebutuhan atau prioritas.
- Terangkan bagaimana hasil pemetaan dapat digunakan untuk mengidentifikasi lokasi yang perlu mendapat perhatian.
- Pastikan angka 330 merujuk pada unit pemetaan yang benar menurut metodologi proyek.

**B. Prioritas retrofit di Q1**

- Jelaskan arti Q1 sesuai definisi analisis yang digunakan.
- Uraikan mengapa kelompok ini layak dipertimbangkan sebagai prioritas.
- Hubungkan temuan dengan usulan tindakan, bukan sekadar menampilkan peringkat atau visualisasi.
- Jangan mengasumsikan bahwa Q1 otomatis merupakan kelompok dengan manfaat ekonomi terbesar atau dampak lingkungan terbesar kecuali terdapat bukti yang mendukungnya.

**C. Penilaian kapasitas gardu**

- Jelaskan bagaimana analisis dapat membantu menyusun prioritas pemeriksaan kesiapan jaringan listrik.
- Bedakan antara estimasi kebutuhan listrik dan kapasitas gardu yang benar-benar terukur.
- Jika data kapasitas aktual tidak tersedia, posisikan hasil sebagai dasar skrining awal atau rekomendasi evaluasi teknis, bukan sebagai bukti bahwa gardu tertentu telah mencukupi atau tidak mencukupi.

**D. Pilot proyek elektrifikasi bergas**

- Jelaskan logika pemilihan lokasi atau sasaran pilot berdasarkan temuan analisis yang tersedia.
- Hubungkan pilot dengan kebutuhan validasi lapangan, kelayakan infrastruktur, dan evaluasi sebelum perluasan program.
- Jangan mengklaim keberhasilan, penghematan, atau dampak terukur yang belum diuji.

**Aturan narasi yang wajib dipatuhi:**

Gunakan frasa persis **“temuan model mendukung rekomendasi ini”** pada bagian kebijakan publik.

Setiap rekomendasi utama harus memiliki alur argumentasi berikut:

**Temuan analisis → interpretasi yang wajar → rekomendasi kebijakan → langkah implementasi atau validasi.**

Jangan menyamakan prediksi model dengan fakta lapangan yang sudah terverifikasi. Jangan menyatakan bahwa hubungan statistik atau asosiasi yang ditemukan otomatis menunjukkan hubungan sebab-akibat.

Khusus untuk interpretasi SHAP:

- Gunakan SHAP untuk menjelaskan kontribusi fitur terhadap keluaran atau prediksi model sesuai metode yang digunakan.
- Jangan mengklaim bahwa nilai SHAP membuktikan kausalitas mutlak.
- Hindari kalimat seperti “fitur X menyebabkan peningkatan konsumsi energi” jika analisis tidak dirancang untuk mengidentifikasi efek kausal.
- Gunakan formulasi yang lebih tepat, misalnya “fitur X berkontribusi terhadap prediksi model” atau “pola ini mengindikasikan bahwa fitur X relevan dalam estimasi model”.
- Jangan mengubah korelasi, kontribusi prediktif, atau pola spasial menjadi klaim dampak kebijakan yang sudah terbukti.

Pilih visual yang langsung membantu pengambilan keputusan, seperti peta prioritas, diagram kuadran, matriks tindakan, atau alur implementasi. Hindari menampilkan grafik teknis tanpa menjelaskan implikasi kebijakannya.

### BAGIAN V — KETERBATASAN DAN PENUTUP

**08:30–09:30**

Tutup presentasi dengan pengakuan keterbatasan yang jujur, lalu simpulkan nilai praktis analisis.

Akui keterbatasan yang benar-benar relevan, termasuk jika data berikut tidak tersedia:

- Data beban puncak aktual.
- Data kapasitas gardu yang terukur atau terverifikasi.
- Informasi lapangan lain yang dibutuhkan untuk memastikan kesiapan implementasi.

Jangan menyatakan semua keterbatasan tersebut sebagai fakta proyek apabila belum dikonfirmasi dalam materi sumber. Verifikasi terlebih dahulu dan cantumkan hanya yang benar-benar sesuai.

Jelaskan secara singkat konsekuensinya: rekomendasi merupakan dasar penentuan prioritas dan evaluasi lanjutan, bukan pengganti audit teknis atau validasi lapangan.

Akhiri dengan kesimpulan yang:

- Menegaskan kembali urgensi Trilema Energi.
- Menghubungkan hasil analisis dengan tiga rekomendasi utama.
- Mengajak audiens mengambil langkah lanjutan yang konkret dan proporsional.
- Tidak memperkenalkan klaim atau angka baru pada menit terakhir.

Penutup harus terdengar percaya diri, realistis, dan berorientasi pada keputusan.

## 4. Format Output Wajib

Susun hasil dalam format *slide-by-slide*. Untuk **setiap slide**, gunakan empat komponen utama berikut serta informasi durasi sebagai metadata tambahan.

**Slide [nomor] — [rentang waktu dan durasi]**

**Judul Slide**
Tuliskan satu pernyataan kuat yang menyampaikan pesan inti slide. Hindari judul generik seperti “Metodologi”, “Hasil Analisis”, atau “Kesimpulan” apabila dapat diganti dengan judul yang lebih informatif.

**Visual Utama**
Jelaskan visual yang perlu ditampilkan, komposisi diagram atau grafik, elemen yang harus diberi penekanan, serta pesan yang harus ditangkap audiens. Sebutkan jika visual harus menggunakan peta, diagram arsitektur, matriks prioritas, atau grafik yang telah tersedia.

**Isi Teks**
Tuliskan teks yang benar-benar akan muncul di slide. Gunakan kalimat ringkas, kata kunci, dan angka penting. Hindari menyalin seluruh narasi pembicara ke dalam slide. Batasi isi agar dapat dibaca dengan cepat oleh audiens.

**Narasi Pembicara**
Tulis naskah lisan yang siap dibawakan, bukan sekadar poin-poin atau instruksi kepada pembicara. Narasi harus menjelaskan makna visual, menghubungkan bukti dengan kesimpulan, dan mengalir secara alami ke slide berikutnya. Cantumkan estimasi durasi dan pastikan panjang narasi sesuai dengan alokasi waktu.

## 5. Standar Kualitas Presentasi

Terapkan seluruh standar berikut:

1. **Top-down:** tampilkan kesimpulan dan rekomendasi sebelum metodologi.
2. **Satu slide, satu pesan utama:** jangan menumpuk banyak argumen dalam satu slide.
3. **Berbasis bukti:** setiap angka, metrik, dan klaim faktual harus bersumber dari materi proyek.
4. **Konsisten:** jaga ketepatan istilah, angka model, definisi fitur, dan interpretasi hasil dari awal sampai akhir.
5. **Berorientasi kebijakan:** jelaskan apa arti hasil analisis bagi pengambil keputusan dan tindakan apa yang perlu dipertimbangkan.
6. **Teknis secukupnya:** gunakan istilah teknis hanya ketika diperlukan untuk menjelaskan validitas atau kekuatan analisis.
7. **Akurat secara ilmiah:** pisahkan hasil prediksi, asosiasi, interpretasi model, dan klaim kausal.
8. **Efisien:** setiap slide harus memiliki kontribusi yang jelas terhadap argumen utama dan durasi presentasi.
9. **Jujur terhadap data:** jangan mengarang metrik kinerja, angka penghematan, tingkat akurasi, validasi lapangan, atau hasil kebijakan yang tidak tersedia.
10. **Siap dipresentasikan:** gunakan narasi yang alami, percaya diri, mudah dihafal, dan tidak terdengar seperti membaca laporan teknis.

Jika informasi yang diperlukan tidak ditemukan dalam materi sumber, tandai dengan format **[PERLU KONFIRMASI: informasi yang dibutuhkan]**. Jangan mengisi kekosongan tersebut dengan angka atau asumsi buatan. Jika terdapat konflik antarangka atau definisi, identifikasi masalahnya secara eksplisit.

## 6. Materi yang Tidak Boleh Masuk ke Presentasi Utama

Jangan memasukkan detail teknis ekstrem yang tidak esensial untuk keputusan kebijakan ke dalam durasi utama, termasuk:

- Penurunan matematis atau implementasi mendalam algoritma bobot adaptif `A2xM2xp2`.
- Penjelasan rinci tentang koreksi galat tetangga.
- Derivasi rumus, kode, atau rincian implementasi yang tidak diperlukan untuk memahami rekomendasi.
- Eksperimen tambahan yang tidak mengubah kesimpulan kebijakan.

Pindahkan materi tersebut ke **slide cadangan Q&A** yang terpisah dari presentasi utama.

Slide cadangan dapat mencakup:

1. Mekanisme bobot adaptif `A2xM2xp2`.
2. Mekanisme koreksi galat tetangga.
3. Validasi rekayasa fitur dan mitigasi *data leakage*.
4. Rincian pembagian model: 28 Gas/EUI, 13 Listrik, dan 2 LightGBM terpisah.
5. Interpretasi SHAP beserta batasannya.
6. Keterbatasan data dan rencana validasi lapangan.

Untuk setiap slide cadangan, berikan judul, visual atau bukti yang perlu ditampilkan, dan pertanyaan juri yang mungkin dijawab oleh slide tersebut. Materi cadangan tidak boleh mengubah alokasi waktu 9 menit 30 detik.

## 7. Urutan Pengerjaan dan Pemeriksaan Akhir

Sebelum menyusun slide, identifikasi terlebih dahulu:

- Tesis utama presentasi.
- Tiga rekomendasi kebijakan yang akan diumumkan pada pembukaan.
- Bukti terkuat yang mendukung setiap rekomendasi.
- Angka dan definisi teknis yang harus dipertahankan.
- Keterbatasan data yang harus diakui.
- Detail teknis yang seharusnya disimpan untuk Q&A.

Kemudian susun slide satu per satu mengikuti alur dan batas waktu yang telah ditetapkan.

Sebelum menyelesaikan jawaban, lakukan pemeriksaan akhir dan tampilkan hasilnya dalam daftar verifikasi singkat:

- [ ] Total durasi bicara tepat 9 menit 30 detik.
- [ ] Buffer 30 detik berada di luar narasi utama.
- [ ] Pembukaan mengungkapkan Trilema Energi dan tiga rekomendasi dalam 60 detik.
- [ ] Pembahasan mengikuti urutan rekayasa fitur, arsitektur model, kebijakan, lalu keterbatasan.
- [ ] Total 43 model dibedakan secara jelas menjadi 41 model yang di-*blend* dan 2 model LightGBM terpisah.
- [ ] Peta 330 zona, prioritas Q1, kapasitas gardu, dan pilot elektrifikasi bergas dibahas sesuai bukti yang tersedia.
- [ ] Frasa “temuan model mendukung rekomendasi ini” digunakan pada bagian kebijakan publik.
- [ ] Tidak ada klaim kausalitas mutlak berdasarkan SHAP.
- [ ] Tidak ada angka atau hasil empiris yang dikarang.
- [ ] Detail teknis ekstrem hanya muncul di slide cadangan Q&A.

**Instruksi final:** Prioritaskan kejelasan argumen, ketepatan teknis, dan kekuatan rekomendasi kebijakan. Jangan sekadar meringkas atau memindahkan susunan slide lama. Rancang ulang seluruh alur agar audiens memahami keputusan yang diusulkan sebelum mendalami bagaimana model dibangun.

Mulailah dengan memberikan **struktur slide-by-slide yang lengkap, narasi pembicara, estimasi durasi setiap slide, slide cadangan Q&A, dan daftar verifikasi akhir** berdasarkan materi sumber yang tersedia.

Simpan ke Dicefest/deck/ppt02
