# CLAUDE.md: DICE FEST 2026 ML (Urban Energy Transition Trilemma)

Aturan inti proyek. Rencana kerja lengkap per fase ada di `MASTER_PROMPT.md`. Baca file itu seluruhnya sekali di awal proyek; setelahnya cukup baca bagian fase yang aktif.

## Misi
Submisi dengan Column-Wise NRMSE serendah mungkin di Private Leaderboard Kaggle `dicefest-2026-machine-learning`, dari pipeline yang reproducible bit-per-bit, bebas kebocoran, dan setiap langkahnya tercatat.

## Setiap sesi dimulai, di-resume, atau setelah compaction
1. Baca `reports/STATUS.md`, lalu `reports/LEADERBOARD.md`, lalu 40 baris terakhir `reports/JOURNAL.md`.
2. Cek proses yang masih jalan: `ps aux | grep -E "python|queue_worker" | grep -v grep`, `nvidia-smi`, isi `queue/*/running/`.
3. Baca bagian fase aktif di `MASTER_PROMPT.md`.
4. `experiments/registry.csv` adalah sumber kebenaran. Jangan mengulang eksperimen yang sudah tercatat kecuali ada alasan tertulis.

Saat compaction, pertahankan: fase aktif, exp_id dan skor terbaik per target, job yang sedang jalan (PID, log), keputusan terbaru, dan 5 langkah berikutnya.

## Fakta kompetisi
- Target, urutan wajib: `target_macro_gas_kwh`, `target_macro_elec_kwh`, `target_eui_gas_kwh_per_sqm`, `target_clean_building_ratio`, `target_electric_heating_adoption`.
- Metrik: rata-rata RMSE_k / sigma_k. Sigma beku: 151750.04, 41725.37, 40.65, 12.49, 7.18. Lebih kecil lebih baik. Prediksi rata-rata konstan memberi sekitar 1.0 per target.
- Metrik terpisah per kolom, jadi setiap target boleh punya pipeline terbaiknya sendiri. Evaluasi selalu di skala asli.
- Submisi: `id` + 5 target, 183.987 baris, urutan `id` identik dengan `sample_submission.csv`. Target persen di-clip 0 sampai 100, target makro dan EUI tidak negatif.
- Limit 5 submisi per hari (reset 07:00 WIB), pilih 2 final. Deadline Kaggle + PDF writeup + notebook .ipynb: 23 Oktober 2026 23:59 WIB.

## Aturan yang tidak boleh dilanggar
1. Dilarang data eksternal, dilarang API LLM untuk menghasilkan nilai prediksi, dilarang hand-labeling. Semua nilai prediksi keluar dari kode open-source yang dijalankan lokal.
2. PENTING: setiap statistik yang memakai target (target encoding, fitur KNN-target, kalibrasi, bobot ensemble, stacking, pseudo-label) dihitung out-of-fold, nested bila perlu. Tidak ada fit yang melihat label baris validasi.
3. Semua eksperimen memakai file fold di `data/folds/`. File fold tidak diubah setelah dibuat; skema baru berarti file baru.
4. Seed dikunci di semua library. Tidak ada budget berbasis waktu (`time_limit`) di pipeline final. Urutan kolom selalu dari list yang di-sort, bukan dari `set()`.
5. Kolom `id` tidak dipakai sebagai fitur.
6. Jangan menimpa `data/raw/`, jangan menghapus `experiments/`, jangan `git reset --hard` atau `git clean`.
7. Tidak ada hasil tanpa catatan (protokol di bawah).

## Cara kerja
- Otonom. Jangan berhenti untuk minta konfirmasi. Keputusan yang bisa diambil dari data: ambil, catat di `reports/DECISIONS.md`, lanjut. Yang benar-benar butuh user: tulis di `reports/NEEDS_USER.md`, lalu kerjakan hal lain.
- Job lebih dari sekitar 5 menit dijalankan di background (`nohup` atau queue worker) dengan log ke file, dipantau dengan `tail -n 30`. Selama menunggu, siapkan eksperimen berikutnya. Jika memang harus menunggu, `sleep` paling lama 9 menit per perintah lalu cek lagi.
- Selalu jalankan `--smoke` (subsample kecil, 2 fold, sedikit iterasi) sebelum run penuh.
- Jangan menebak API library dari ingatan. Cek versi terpasang dan `help()`, `inspect.signature`, atau dokumentasi versinya.
- Jaga konteks: jangan print DataFrame lebar atau log panjang. Tulis detail ke file, tampilkan ringkasan.
- Resource: RAM 24 GB, VRAM 16 GB. Satu job GPU berat pada satu waktu. Job CPU boleh paralel jika RAM cukup. Cek `free -g` dan `nvidia-smi` sebelum meluncurkan job.
- Cek waktu dengan `TZ=Asia/Jakarta date` di awal setiap langkah besar dan patuhi gerbang jadwal (MASTER_PROMPT bagian 17).
- Sejak baseline pertama, `submissions/BEST_CURRENT.csv` harus selalu ada dan lolos `scripts/validate_submission.py`.
- Di gerbang penting (setelah desain CV, sebelum freeze final), minta subagent dengan konteks bersih mengaudit kode terhadap kebocoran dan determinisme.
- Di setiap gerbang dan akhir fase, tampilkan ringkasan maksimal 15 baris di percakapan.

## Protokol pencatatan (setiap eksperimen)
1. Sebelum run: tulis hipotesis di `reports/JOURNAL.md` dan buat config di `configs/`.
2. Run menulis ke `experiments/<EXP_ID>_<slug>/`: config.yaml, run.log, metrics.json, oof.parquet, test_pred.parquet, feature_importance.csv, env.json, README.md.
3. Satu baris di `experiments/registry.csv` (running, lalu done atau failed).
4. Jalankan `python scripts/update_reports.py` (LEADERBOARD.md + dashboard.html).
5. Tulis kesimpulan di JOURNAL.md, perbarui STATUS.md, BACKLOG.md, dan COVERAGE.md.
6. `git add -A && git commit -m "<EXP_ID>: <ringkas> | CV <skor>"` (data dan artefak besar ada di .gitignore).

## Gaya
- Laporan dalam Bahasa Indonesia. Kode, nama variabel, dan komentar dalam bahasa Inggris.
- Draf writeup dan bahan presentasi: Bahasa Indonesia yang natural, tidak kaku, tidak terdengar seperti teks mesin, tanpa karakter em-dash (U+2014).
