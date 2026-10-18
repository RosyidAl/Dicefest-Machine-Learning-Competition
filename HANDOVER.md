# HANDOVER: DICE FEST 2026 Machine Learning (tim BISMILLAH)

Dokumen serah terima untuk agent Claude berikutnya. Ditulis 18 Okt 2026 sekitar 19:00 WIB, sebelum user memigrasi akun Claude.
Baca dokumen ini sampai habis sebelum menyentuh apa pun. Setelah itu baca `CLAUDE.md` (aturan inti, wajib dipatuhi), lalu `reports/STATUS.md` bagian HANDOFF. `MASTER_PROMPT.md` adalah rencana kerja lengkap per fase; cukup baca bagian yang relevan (Fase 11 audit reproducibility, Fase 12 laporan).

---

## 0. Ringkasan dalam 10 baris

1. Kompetisi Kaggle `dicefest-2026-machine-learning`: memprediksi 5 target energi kawasan permukiman; metrik rata-rata RMSE/sigma per kolom (lebih kecil lebih baik).
2. Model final yang dipilih: **v19** (ensemble 39 anggota: XGBoost, LightGBM, beberapa varian MLP) + koreksi residual + model adopsi + salin baris duplikat.
3. **Public LB terbaik: 0,24934** (sub_v19dev). Peringkat public 6 (snapshot 18 Okt). Prediksi CV-ke-LB kita akurat sampai digit kelima di tiga versi terakhir.
4. Pengembangan model praktis jenuh: gain per langkah tinggal 0,00002 sampai 0,00007. Model tidak perlu diubah lagi kecuali user meminta.
5. Yang **belum** selesai (butuh server sekitar 21 jam): run pipeline final untuk 10 anggota baru, uji drift + sha256, artefak, eksekusi notebook headless, notebook cadangan dari artefak.
6. Yang juga belum: angka akhir di writeup/laporan/presentasi, build PDF, keputusan D-FINAL, checklist user.
7. **Saat serah terima tidak ada job yang berjalan.** User sedang memakai server untuk proyek lain di `/root/kaggle`. **Jangan jalankan job apa pun sebelum user mengizinkan.**
8. **Blocker disk:** hanya sekitar 12 GB bebas. Notebook final butuh sekitar 25 GB (model 40 anggota). Minta user membebaskan ruang dulu (lihat bagian 8).
9. Deadline: **Jumat 23 Okt 2026 23:59 WIB** (submisi Kaggle + PDF writeup + notebook .ipynb lewat dicefest.web.id). Pekerjaan server harus mulai paling lambat **Selasa 20 Okt malam**.
10. Setelah deadline: 24 Okt audit kode, 25 Okt pengumuman Top 10 (murni dari private LB), 30 Okt presentasi final Zoom (10 menit paparan + 10 menit tanya jawab), 31 Okt pengumuman juara.

---

## 1. Tentang user dan cara bekerja dengannya

- User berkomunikasi dalam **Bahasa Indonesia**. Balas dalam Bahasa Indonesia, ringkas, dengan angka. Kode, nama variabel, dan komentar kode dalam bahasa Inggris.
- User **submit ke Kaggle sendiri** (tidak ada kredensial Kaggle di server) lalu melaporkan skornya. Kita menyiapkan file dan memberi tahu file mana yang di-submit.
- User **tidak suka pop-up AskUserQuestion**. Ajukan pertanyaan sebagai teks biasa di balasan.
- User ingin **hasil maksimal walau gain kecil** ("pokoknya hasil terbaik"). Tetapi sekarang prioritasnya menyelesaikan paket final yang reproducible.
- User **dua kali meminta memakai data eksternal yang disamarkan** agar lolos audit (termasuk skema "satu final ori, satu final curang"). Permintaan itu ditolak dan **tetap harus ditolak**: melanggar aturan (data eksternal dilarang), dan aturan mewajibkan notebook yang dikumpulkan mereproduksi file submisi pemenang. Tawarkan jalan sah sebagai gantinya.
- Server dipakai bersama proyek lain milik user di `/root/kaggle` (job `scripts/train_noisy_student.py`, BirdCLEF). **Jangan menyentuh, menjeda, atau menghentikan job itu tanpa izin eksplisit user.** Izin yang pernah diberikan (17 Okt) tidak berlaku lagi.
- User bisa menutup laptop; sesi Claude bisa terputus. Semua job panjang harus berupa unit systemd (lihat bagian 9), bukan proses milik shell.

---

## 2. Fakta kompetisi

| Hal | Isi |
|---|---|
| Target (urutan wajib) | `target_macro_gas_kwh`, `target_macro_elec_kwh`, `target_eui_gas_kwh_per_sqm`, `target_clean_building_ratio`, `target_electric_heating_adoption` |
| Sigma beku | 151750.04, 41725.37, 40.65, 12.49, 7.18 |
| Metrik | rata-rata RMSE_k / sigma_k di skala asli; prediksi konstan sekitar 1,0 per target |
| Data | train 429.303 baris, test 183.987 baris; 201 fitur anonim (bld_dim, bld_era, hvac_dist, env_climate, socio_demog) + `zone_region_id` (330) + `subzone_id` (sekitar 150 ribu) |
| Split | Public LB 30% test, Private LB 70% test; finalis (Top 10) murni dari private LB |
| Submisi | `id` + 5 target, 183.987 baris, urutan id = sample_submission; maksimal 5 per hari (reset 07:00 WIB); pilih maksimal 2 final |
| Aturan | Hanya data resmi. Dilarang data eksternal, API LLM untuk nilai prediksi, hand-labeling. Seed dikunci; output harus identik saat diaudit. Notebook .ipynb wajib mereproduksi file submisi pemenang secara deterministik. |
| Tim | **BISMILLAH** (nama notebook: `final/DICEFEST2026_BISMILLAH.ipynb`) |

---

## 3. Temuan data yang menentukan pendekatan

1. **Identitas eksak.**
   - `clean_building_ratio = round(100 * bld_dim_02 / bld_dim_01, 2)` di 100% baris: dihitung, bukan diprediksi.
   - `EUI x bld_dim_06 = gas` sampai presisi mesin: `bld_dim_06` adalah luas lantai A. Semua model keluarga gas memprediksi EUI (atau laju per eksposur) lalu dikalikan kembali; bobot blend di-fit terpisah untuk metrik gas dan EUI.
   - Adopsi pemanas listrik konstan per subzona (unit sensus OA). Subzona yang ada di train: nilainya disalin (92% baris test). Subzona baru: model binomial (`adopt_v2`, jumlah rumah tangga N direkonstruksi eksak dari pecahan `hvac_dist`).
2. **Split test acak per baris** (bukti: rasio train 0,700000; baris test di subzona baru 8,14% vs harapan 8,17%; adversarial AUC 0,50). Konsekuensinya, KFold acak per baris meniru test dan target encoding subzona/zona sah dipakai (out-of-fold).
3. **Noise label besar:** varians residual log = a + b/M (M = jumlah meter). Pada blend final 73% (gas) dan 77% (listrik) varians residual adalah noise rata-rata rumah tangga. Lantai skor dari fitur saja sekitar 0,21.
4. `sub_n` (nomor subzona) berurutan secara spasial; `env_climate_01/03/05` mendefinisikan sel grid iklim.
5. 77 baris test identik dengan baris train pada kunci (socio_demog_005, subzona, M, E, A); targetnya disalin (`dup_fix`).
6. Porsi galat kita: EUI sekitar 52%, listrik 25%, gas 20%, adopsi 3%, rasio bersih 0.

Rincian: `reports/eda/EDA_REPORT.md`, memori `~/.claude/projects/-root-Dicefest/memory/dicefest-data-findings.md`.

---

## 4. Arsitektur solusi (v19)

**Formulasi target:** laju terhadap eksposur (gas per meter `gpm`, per luas `gas_per:bld_dim_06`, per luas selubung `gas_per:bld_dim_03`, per sertifikat `gas_per:bld_dim_01`, log, listrik per meter `epm`, per luas `epa`, dll.) dengan bobot sampel yang membuat loss setara MSE skala asli. Sebagian anggota memakai objektif Poisson (GLM laju dengan eksposur).

**Fitur:** fitur dasar + rasio fisik; agregat transduktif per subzona/zona (train + test, tanpa target); target encoding out-of-fold (subzona, zona, jendela sub_n ±2/±10, sel iklim di luar subzona sendiri); **FB07** kNN-target (rata-rata target tetangga paling mirip stok bangunannya dalam jendela sub_n 30/100/300); **FB07b** blok kNN-target tambahan (ruang era, bangunan + PCA sensus, ukuran; jendela W=10 dan W=1000). Semua fitur bertarget cross-fitted.

**Model:** 39 anggota blend (27 keluarga gas/EUI, 12 listrik), 40 entri di `members.yaml` (mk1_gpm ikut dilatih bersama mk1_epm). Keluarga: XGBoost CUDA (deterministik), LightGBM CPU (deterministic=True), MLP polos, MLP + embedding PLR, TabM-mini, MLP multi-task gas+listrik. Mayoritas 10-fold, KFold(shuffle, seed 42).

Anggota blend v19:
- gas/EUI: xa_gpm, p1_gpm, lgb_gas, x8_gpn, n6_lgpm, xgb_eui_wsM, la_gas, xgb_leui_wA, xgb_gpn_tew, n4f_gpm, x8_lgpm, xgb_gas, xk_gpgea, mk2_gpm, pk2_gpgea, lp_gpm, xk2_gpgea, lk2_gpm, xp_gpgea, xp_eui, xp8_gpm, xp_gpm, xp82_gpm, xp2_gpgea, lp2_gpm, pk2_lgpm, xp2_eui
- listrik: t1_lepm, l4_epa, la_lepa, n6b_lepm, xgb_elec, mk1_epm, lp_epm, mk2_epm, pk2_lepm, xp_epa, lp2_epm, xp2_epa

**Ensemble:** NNLS per metrik (gas dengan skala baris A; EUI; listrik), dinilai nested (bobot di-fit di fold lain). Seleksi anggota greedy maju-mundur.

**Post-processing:**
- Koreksi residual: rata-rata residual OOF yang di-shrink dari zona, sel iklim, jendela ±30 sub_n, dan 5-6 konfigurasi kNN kemiripan per metrik. Bobot λ di-fit di holdout internal P70/H30 (seed 7), **tidak pernah di OOF penuh** (bias lewat TE). λ v19 = `exp/resid_knn_v19.json` = `final/configs/resid_knn.json`.
- Adopsi: salin nilai train; subzona baru dengan `adopt_v2`.
- Duplikat: 77 baris disalin.
- Determinisme: tahap assemble memakai `threadpool_limits(1)` (D008), karena OpenBLAS multithread mengubah bit terakhir sesuai jumlah thread.

---

## 5. Riwayat progres dan hasil

**Public LB per versi** (CV = blend nested 5 target sebelum post-processing):

| Tanggal | Versi | CV nested | Public LB | Perubahan utama |
|---|---|---|---|---|
| 13 Okt | probe x4 | | 0,25317 | XGB tunggal |
| 13 Okt | v6pre | 0,25075 | 0,25125 | blend NNLS formulasi rasio |
| 14 Okt | v8 / v8nc | 0,25061 / 0,25066 | 0,25116 / 0,25118 | XGB d10 lr .03; tanpa CatBoost |
| 14 Okt | v9r | 0,25055 | 0,25070 | PLR MLP + koreksi residual grup |
| 15 Okt | v12r | 0,25015 | 0,25025 | agregat transduktif, PLR/TabM |
| 15 Okt | v13ra | 0,25007 | 0,25008 | model adopsi binomial |
| 15 Okt | v15rd / v15ncrd | 0,25001 / 0,25002 | 0,25000 / 0,2499 | MLP multi-task, salin duplikat |
| 16 Okt | v15ncrkd | 0,25002 | 0,24954 | koreksi kNN kemiripan |
| 16 Okt | v16dev | 0,24960 | 0,24943 | fitur kNN-target FB07 di dalam model |
| 18 Okt | v18dev | 0,24940 | 0,24936 | anggota FB07b + Poisson |
| 18 Okt | **v19dev** | **0,24937** | **0,24934** | Poisson x FB07b, pk2_lgpm |

Public LB selalu = CV + estimasi holdout post-processing + sekitar 0,0005 (sd 0,00004, 10 submisi). Prediksi rantai holdout v18 (-0,00006) dan v19 (-0,00002) tepat di LB.

**Skor per target v19** (CV nested): gas 0,24498, listrik 0,31589, EUI 0,65029, rasio bersih 0, adopsi sekitar 0,0357.
**Rantai holdout v19** (blend + koreksi): gas 0,24531, EUI 0,64930, listrik 0,31475 (v18: 0,24533 / 0,64937 / 0,31476; v16: 0,24539 / 0,64961 / 0,31477).

**Yang sudah dicoba dan tidak berhasil** (tercatat di registry/journal): stacking level-2, blend per ukuran, bobot dari holdout, blend khusus subzona baru, CatBoost di final (tidak deterministik), loss Poisson pada MLP, koreksi residual lintas target, adopsi sebagai fitur, bobot koreksi per ukuran atau diskalakan untuk kepadatan, 20 fold, pencarian kebocoran (urutan ID, nilai gas kembar, rasio antar kolom, copula). Pengecekan data eksternal tidak dilakukan dan tidak boleh.

**Keputusan penting** (`reports/DECISIONS.md`): D001 split acak per baris; D002 KFold per baris; D004 determinisme per keluarga (CatBoost dibuang); D006 freeze v16; D007 Poisson saja ditolak; D008 BLAS satu thread di assemble; D009 v18; D010 v19 = kandidat Final A.

---

## 6. Status saat serah terima (18 Okt sekitar 19:00 WIB)

- **Tidak ada job DICE FEST yang berjalan.** `systemctl list-units 'dice-*'` kosong. Job user di `/root/kaggle` sedang berjalan (GPU dan RAM terpakai olehnya).
- **File submisi:**

  | File | sha256 | Arti |
  |---|---|---|
  | `submissions/sub_v19dev.csv` | `62444504fb7de677d10fd69d17b06da796076b75e96d103cef2d22dd95dff216` | v19, LB 0,24934, **acuan untuk notebook final** |
  | `submissions/sub_v18dev.csv` | `07f67f8cda7d5d07200a08cf49c5547b3556fa1b47183e467d999fbf8dbf135b` | v18, LB 0,24936 |
  | `submissions/sub_v16fin.csv` | `3767ba10a82e28d1cf80bf2b62debb20f612fa8d10d2f0c929488e0e481f59be` | v16 dari pipeline final (run #1, kode D008), skor = 0,24943 |
  | `submissions/BEST_CURRENT.csv` | = sub_v19dev | |

- **Pipeline final** (`final/`): kode sudah mendukung v19.
  - `final/src/pipeline.py`: `make_data` (satu DataX per jalur, FB07b aktif per anggota lewat `knn_x: true`), assemble dengan BLAS satu thread.
  - `final/src/fb07b.py`: THREADS=3 untuk menekan RAM (dev memakai 8; nilai tidak bergantung jumlah thread, BLOCK=20000 harus tetap).
  - `final/configs/members.yaml`: v19 (40 entri); `resid_knn.json`: λ v19.
  - Cadangan: `members_v18.yaml` / `resid_knn_v18.json`, `members_v16.yaml` / `resid_knn_v16.json`.
  - `final/make_notebooks.py`: sudah menanam `features_knn.py` dan `fb07b.py` (bug lama: notebook v16 akan gagal di anggota kNN pertama, sudah diperbaiki). Notebook **belum** dibangun ulang untuk BISMILLAH dan **belum** dieksekusi.
- **`final/work/run1`** (23 GB): output 48 anggota dari run final v16 + v18, termasuk model per fold dan cache kNN. **Belum ada** 10 anggota: lk2_gpm, lp_gpm, lp_epm, xp82_gpm, xp2_gpgea, xp2_eui, xp2_epa, lp2_gpm, lp2_epm, pk2_lgpm.
  - Run final v16 terbukti identik byte dengan array pengembangan untuk 35 anggota.
  - Untuk anggota v18 di run1, baru skor mk2 yang dicek identik. Uji drift lengkap belum dilakukan.
- **Dokumen Fase 12 (draf, angka perlu diperbarui ke v19):** `reports/WRITEUP_DRAFT.md` (8 halaman PDF, 8 figur), `reports/PRESENTATION_OUTLINE.md` (9 slide + 14 tanya jawab), `reports/FINAL_REPORT.md`, `reports/figures/` (fig1-fig5, shap_*), `final/README_REPRODUCE.md`.
- **Disk:** sekitar 12 GB bebas dari 148 GB. Pemakai terbesar: `/root/kaggle` 62 GB (proyek user), `/root/.cache/huggingface` 12 GB (proyek user), `final/work/run1` 23 GB.

---

## 7. Langkah berikutnya (urut, dengan perintah)

Jalankan hanya setelah user mengizinkan memakai server. Sebelum setiap peluncuran: `free -g`, `nvidia-smi`, `df -h /root`, dan `systemctl list-units 'dice-*'`.

### Langkah 1: run final v19 (sekitar 4-5 jam)
Resume di `final/work/run1`: hanya 10 anggota yang belum ada yang dilatih, lalu adopsi + assemble.
```bash
cd /root/Dicefest
systemctl reset-failed dice-run1 2>/dev/null
systemd-run --unit=dice-run1 --working-directory=/root/Dicefest -E HOME=/root \
  -p StandardOutput=append:/root/Dicefest/logs/run1.log -p StandardError=append:/root/Dicefest/logs/run1.log \
  .venv/bin/python -u final/run_all.py --work final/work/run1 --out final/output/submission_run1_v19.csv
```
Pantau `final/work/run1/lane_{xgb,nn,cpu}.log` dan `run_all.log`. Jalur CPU (5 LightGBM) yang paling lama.
Kalau RAM kritis (MemAvailable < 3 GB), hentikan **satu proses jalur** secara manual dengan `kill <pid>`. Jangan biarkan OOM killer: systemd akan menghentikan seluruh unit. Lalu resume lagi setelah jalur lain selesai.

### Langkah 2: uji drift dan sha256
```bash
.venv/bin/python - <<'EOF'
import os, numpy as np
run = "final/work/run1"
bad = [(n, p) for n in sorted(f[:-4] for f in os.listdir(f"{run}/pred")) for p in ("oof", "pred")
       if os.path.exists(f"exp/{p}/{n}.npy") and not np.array_equal(np.load(f"{run}/{p}/{n}.npy"), np.load(f"exp/{p}/{n}.npy"))]
print("differing arrays:", bad)
EOF
sha256sum final/output/submission_run1_v19.csv   # harus 62444504...f216 (= sub_v19dev)
```
Kalau ada array berbeda, cari penyebabnya sebelum lanjut. Contoh penyebab: tie jarak di kNN, BLOCK/urutan berbeda, kode dev dan final tidak sama. Kalau sha256 cocok, catat di JOURNAL + DECISIONS dan perbarui `final/README_REPRODUCE.md`.

### Langkah 3: artefak dan notebook (sekitar 16-20 jam)
**Syarat disk:** butuh minimal sekitar 30 GB bebas. Minta user membersihkan ruang di `/root/kaggle` atau `/root/.cache/huggingface` (milik user, jangan dihapus tanpa izin). Yang boleh kita hapus sendiri hanya sekitar 1,5 GB: `cache/knn_fb07b_*`, `final/work/drift`, `exp/fb07b`.
```bash
bash scripts/make_artifacts.sh final/work/run1          # mv models -> final/artifacts + SHA256SUMS (tanpa salin)
.venv/bin/python final/make_notebooks.py BISMILLAH       # -> final/DICEFEST2026_BISMILLAH.ipynb + inference_from_artifacts.ipynb
systemd-run --unit=dice-nbfull --working-directory=/root/Dicefest -E HOME=/root \
  -p StandardOutput=append:/root/Dicefest/logs/nb_full.log -p StandardError=append:/root/Dicefest/logs/nb_full.log \
  /bin/bash scripts/run_notebook.sh final/DICEFEST2026_BISMILLAH.ipynb final/nbrun_full
systemd-run --unit=dice-nbinf --working-directory=/root/Dicefest -E HOME=/root \
  -p StandardOutput=append:/root/Dicefest/logs/nb_inf.log -p StandardError=append:/root/Dicefest/logs/nb_inf.log \
  /bin/bash scripts/run_notebook.sh final/inference_from_artifacts.ipynb final/nbrun_inf
```
- Setiap notebook dijalankan di folder sendiri, karena notebook penuh menghapus `dicefest_nb/work` miliknya.
- Kedua notebook harus memberi sha256 = 62444504...f216. Notebook inference ringan (sekitar 2-3 jam) dan boleh jalan bersamaan dengan notebook penuh kalau RAM cukup.
- Hapus file `final/DICEFEST2026_ISI_NAMA_TIM.ipynb` yang lama (placeholder nama tim). Commit notebook BISMILLAH beserta versi tereksekusinya.

### Langkah 4: dokumen dan paket final (bisa tanpa GPU)
- `reports/WRITEUP_DRAFT.md`: isi `[[TEAM_NAME]]` = BISMILLAH, ganti angka v16 ke v19, isi `[[SHA]]` dan `[[jam]]`. Build PDF dari venv terpisah, jangan memasang paket ke `.venv`:
  `python3 -m venv <scratch>/pdfenv && <scratch>/pdfenv/bin/pip install xhtml2pdf markdown-it-py pymupdf && <scratch>/pdfenv/bin/python scripts/build_writeup_pdf.py reports/WRITEUP_BISMILLAH.pdf`
- Regenerasi figur bila perlu (`src/make_figures.py`; tambahkan baris v18/v19 di `reports/cv_lb_history.csv`; komposisi blend ambil dari `exp/v19_dev_work/summary.json`).
- Perbarui `reports/FINAL_REPORT.md`, `reports/PRESENTATION_OUTLINE.md`, `final/README_REPRODUCE.md` (waktu jalan per tahap, sha256 yang diharapkan).
- Tulis **D-FINAL** di DECISIONS dan **checklist akhir** di `reports/NEEDS_USER.md` (lihat bagian 10).
- Jalankan `scripts/update_reports.py`, perbarui STATUS/COVERAGE, lalu commit.

### Langkah 5 (opsional, hanya kalau user meminta dan waktu cukup): v20
Kandidat sudah didefinisikan di `src/exp_v18_knn2.py`: versi kNN + FB07b dari anggota lama tanpa kNN (xk2_gpn, xk2_eui_wsM, xk2_gas, xk2_leui_wA, xk2_elec, lk2_gas, lk2_leui_wA, lk2_epa, nk2_gpm, nk2_lgpm, nk2b_lepm, pk2_gpm, tk2_lepm).
- Perkiraan gain total -0,00002 sampai -0,00005 (polanya menyusut: v18 -0,00007, v19 -0,00002).
- Biaya: full + holdout semua kandidat (sekitar 10-12 jam), seleksi, fit λ, update pipeline final, run final anggota baru, lalu **notebook harus diulang** (sekitar 16-20 jam).
- Hanya layak bila masih ada minimal sekitar 3 hari sebelum deadline dan server tersedia penuh. Pada 18 Okt rekomendasinya: **lewati**.

---

## 8. Prediksi hasil

- **Public LB final:** 0,24934 (= sub_v19dev, karena notebook final harus identik byte).
- **Private LB:** sekitar 0,249, dengan noise sekitar ±0,0005 terhadap public. Noise ini berlaku untuk semua tim, jadi urutan di klaster yang rapat bisa bergeser.
- **Snapshot public LB 18 Okt:**

  | # | Tim | Skor |
  |---|---|---|
  | 1 | Internkc 6767 | 0,04644 |
  | 2 | Neu Automobile | 0,23666 |
  | 3 | Il Principe | 0,23857 |
  | 4 | Hbat | 0,24809 |
  | 5 | opo yo | 0,24897 |
  | 6 | **BISMILLAH (kita)** | **0,24934** |
  | 7 | We are him | 0,24963 |
  | 8 | Tembok Ratapan Reborn | 0,25013 |
  | 9 | salt bread | 0,25030 |
  | 10 | Tim Coba-coba | 0,25153 |

- **Interpretasi:**
  - Skor 0,046 / 0,237 / 0,239 menurut analisis kita tidak bisa dicapai dari data resmi (lantai fitur sekitar 0,21; butuh nilai hampir persis per kawasan). Polanya cocok dengan pemakaian data publik Inggris (DESNZ, sensus), yang dilarang dan kemungkinan tersaring audit 24 Okt.
  - **Top 10 hampir pasti aman**: selisih ke posisi 10 sekitar 0,002, empat kali noise private.
  - Di antara tim yang tampak legit, kita sekitar peringkat 3 (Hbat 0,24809, opo yo 0,24897, kita 0,24934, We are him 0,24963). Selisih dengan opo yo dan We are him hanya beberapa per sepuluh ribu, sehingga bisa berubah di private.
  - Juara ditentukan di presentasi 30 Okt. Kekuatan kita di sana: validasi yang terbukti akurat terhadap LB, pipeline reproducible bit per bit, eksperimen tercatat lengkap, dan insight kebijakan dari SHAP.

---

## 9. Lingkungan dan cara menjalankan job

- Hardware: NVIDIA RTX A4000 16 GB, Intel Xeon Silver 4216 16 vCPU, RAM 32 GB, disk 148 GB, Ubuntu 26.04. Container di server kampus; user mengakses lewat code-server.
- Python: **selalu `.venv/bin/python`**. Versi kunci: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, torch 2.14.1+cu130, threadpoolctl 3.7.0 (`final/requirements.txt`).
- **Job panjang wajib lewat systemd-run.** Proses `nohup setsid` pernah terbunuh serentak saat klien code-server terputus. Pola:
  `systemd-run --unit=dice-<nama> --working-directory=/root/Dicefest -E HOME=/root -p StandardOutput=append:<log> -p StandardError=append:<log> <perintah>`
  Untuk job pengembangan tambahkan `-p OOMScoreAdjust=500` (agar job final terlindungi) dan pakai `logs/dev_chain.sh <guard_log> script:mode:nama ...`. Skrip itu berhenti bila MemAvailable < 3 GB.
- Cek job: `systemctl list-units 'dice-*' --all`, `systemctl status <unit>`, log di `logs/`.
- Satu job kira-kira butuh RAM 6-9 GB (Data + TE + kNN + FB07b) dan VRAM 2-4 GB (XGB) atau 3-8 GB (NN; TabM bisa sampai 7,7 GB). Maksimal sekitar 3 job berat bersamaan.
- Pemantau latar Claude Code (shell background) bisa dihentikan harness saat RAM menipis. Itu tidak memengaruhi unit systemd.

---

## 10. Rencana pemilihan final (D-FINAL, belum ditulis)

- **Final A = output notebook v19** (sha256 harus = sub_v19dev). Kalau cocok, user cukup memilih submisi sub_v19dev yang sudah ada di Kaggle (LB 0,24934).
- **Final B:** aturan proyek (MASTER_PROMPT Fase 10) mewajibkan file final berasal dari notebook final, dan notebook kita hanya mereproduksi v19. Rekomendasi: **pilih satu final saja (A)**. Kalau user ingin dua, satu-satunya B yang sah adalah varian yang juga dihasilkan notebook yang sama. Saat ini belum ada; membuatnya berarti mengubah kode dan menjalankan ulang notebook, jadi tidak disarankan.
- **Checklist untuk user** (tulis di `reports/NEEDS_USER.md`):
  1. Pilih final A di Kaggle sebelum 23 Okt 23:59.
  2. Unggah `final/DICEFEST2026_BISMILLAH.ipynb` (versi tereksekusi) dan PDF writeup lewat dicefest.web.id.
  3. Cek ketentuan format writeup di guidebook (tidak bisa diakses dari server).
  4. Simpan `final/artifacts/` untuk audit.

---

## 11. Peta file penting

| Path | Isi |
|---|---|
| `CLAUDE.md`, `MASTER_PROMPT.md` | aturan inti dan rencana kerja |
| `reports/STATUS.md` | status terkini + bagian HANDOFF |
| `reports/JOURNAL.md`, `reports/DECISIONS.md` | log eksperimen dan keputusan (D000-D010) |
| `reports/NEEDS_USER.md`, `reports/COVERAGE.md`, `reports/LEADERBOARD.md` | kebutuhan user, cakupan pendekatan, papan skor internal |
| `experiments/registry.csv` | sumber kebenaran semua eksperimen (E0001-E0280, EM, ENS, PP) |
| `submissions/submission_log.csv` | semua file submisi + sha256 + LB |
| `src/train_lib.py`, `src/nn.py`, `src/fb07b.py`, `src/features_knn.py` | kode training pengembangan (GBDT, MLP, FB07b, kNN) |
| `src/exp_v16_knn.py`, `src/exp_v17_pois.py`, `src/exp_v18_knn2.py` | definisi anggota FB07, Poisson, FB07b + v19/v20 (mode `full` / `hold`) |
| `src/resid_knn.py` | fit λ koreksi residual di holdout (`fit --lam <json>`) |
| `exp/analysis/select_v18.py`, `select_v19.py` | seleksi anggota nested |
| `scripts/assemble_dev.py` | rakit submisi dari array pengembangan dengan kode assemble final |
| `final/run_all.py`, `final/src/`, `final/configs/` | pipeline final (lane xgb/nn/cpu, resume otomatis) |
| `final/make_notebooks.py`, `scripts/run_notebook.sh`, `scripts/make_artifacts.sh` | notebook dan artefak |
| `scripts/build_writeup_pdf.py`, `src/make_figures.py`, `src/interpret.py` | PDF writeup, figur, SHAP |
| `exp/oof`, `exp/pred`, `exp/hold/{oof,pred}` | array OOF/test pengembangan dan holdout semua anggota |
| `exp/v18cfg`, `exp/v19cfg` | konfigurasi blend + λ untuk perakitan dev |
| `logs/v18_guard.log`, `logs/v19_guard.log`, `logs/v20_guard.log`, `logs/run1.log` | status rantai job |

---

## 12. Pelajaran dan jebakan (insiden nyata 16-18 Okt)

1. **RAM:** SIGSTOP tidak membebaskan RAM maupun VRAM. Proses yang dijeda tetap dihitung. Run final pernah terkena OOM (16 dan 17 Okt) karena hal ini.
2. **VRAM:** meluncurkan job GPU saat VRAM bebas hanya sekitar 5 GB menyebabkan OOM GPU yang ikut menjatuhkan run final (17 Okt). Cek `nvidia-smi` sebelum meluncurkan.
3. **FB07b paralel:** tiga jalur `run_all` yang menghitung FB07b bersamaan memakan sampai 29 GB. Sekarang THREADS=3 di kode final; tetap pantau RAM.
4. **Determinisme:** array anggota identik byte, tetapi file submisi sempat berbeda di bit terakhir (24 nilai) karena OpenBLAS multithread. Diperbaiki dengan D008 dan terbukti dengan 1 vs 16 thread.
5. **Notebook:** daftar modul yang ditanam harus mencakup semua import lokal. Ini pernah kurang (features_knn); cek ulang setiap kali ada modul baru.
6. **Jangan** stop atau restart unit systemd yang sedang menjalankan job berharga tanpa memeriksa isinya (mk2 sempat ikut terhenti).
7. **Jangan** pakai `pkill -f` / `pgrep -f` (pernah membunuh shell harness). Kill hanya PID eksplisit dari `ps`.
8. **Waktu:** selalu cek `TZ=Asia/Jakarta date` sebelum menulis timestamp. Agent sebelumnya beberapa kali salah menulis jam dan hari (18 Okt 2026 adalah hari Minggu).
9. `scripts/validate_submission.py` otomatis menambah baris ke submission_log. Pakai `--no-log` bila baris sudah ditulis manual.

## 13. Yang tidak boleh dilakukan

- Data eksternal dalam bentuk apa pun, termasuk disamarkan. Hand-labeling. API LLM untuk nilai prediksi.
- Menimpa `data/raw/`, menghapus `experiments/`, `git reset --hard`, `git clean`, mengubah file fold di `data/folds/`.
- `time_limit` atau batas waktu di pipeline final.
- Menyentuh job atau file di `/root/kaggle` dan `/root/.cache/huggingface` tanpa izin user.
- Menjalankan job DICE FEST saat user sedang memakai server, tanpa izin.
- Mengubah model final (members.yaml / λ) setelah notebook final dieksekusi, kecuali notebook diulang dan sha256 diverifikasi ulang.
