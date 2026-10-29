---
description: Jalankan daftar periksa awal sesi DICE FEST (status, proses berjalan, GPU, fase aktif)
---

Jalankan langkah awal sesi sesuai `CLAUDE.md`:

1. Baca `reports/STATUS.md`, `reports/LEADERBOARD.md`, dan 40 baris terakhir `reports/JOURNAL.md`.
2. Cek proses yang masih jalan: `ps aux | grep -E "python|queue_worker" | grep -v grep`, `nvidia-smi`, `free -g`, dan isi `queue/*/running/`.
3. Cek waktu dengan `TZ=Asia/Jakarta date`.
4. Baca bagian fase aktif di `.claude/MASTER_PROMPT.md`.

Tampilkan ringkasan maksimal 10 baris: fase aktif, skor terbaik per target, job yang sedang jalan (PID dan log), serta tiga langkah berikutnya.
