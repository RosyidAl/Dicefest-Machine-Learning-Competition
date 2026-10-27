# Fakta turunan untuk deck (04_struktur_deck.md)

Angka di deck yang tidak tercetak persis sama di teks atau keluaran notebook: hasil pembulatan, perkalian, bacaan
grafik, atau nilai dari kode dan berkas konfigurasi yang tertanam di notebook. Lembar fakta utama adalah
`fakta_notebook.txt`, yang berisi teks markdown dan keluaran semua sel. Kedua berkas dipakai bersama oleh
`check_deck.py`.

| Angka di deck | Slide | Asal di notebook | Cara mendapatkan |
|---|---|---|---|
| 3,65 TWh | 7 | keluaran sel 49, `hemat_retrofit_TWh` Q1 = 3.650 | dibulatkan |
| 46,8% | 7 | keluaran sel 49, `porsi_gas` Q1 = 0.468 | dikali 100 |
| +4,3 poin; ±35%; ±80%; ±95% | 9 | grafik sel 47, panel kanan atas, garis rata-rata merah | dibaca dari grafik, perkiraan |
| 80% gas untuk pemanasan; efisiensi pompa panas 3 | 7 (narasi) | `explain.py` (sel 19): `HEAT_SHARE, SCOP = 0.8, 3.0`; teks bagian 7.2 | nilai kode |
| ±30, ±100, ±300 nomor subzona | 4 | `train_lib.Data.KNN_CONFIGS` (sel 8): W = 30, 100, 300 | nilai kode |
| 0,453; 0,346; −0,020; 0,037; 0,126; 0,079; 0,072; 0,124; 0,156 | C2 | `configs/resid_knn.json` (sel 22), kunci `lam`, `gas` | dibulatkan 3 desimal |
| n/(n + 30), n/(n + 5) | C2 | `postprocess.py` (sel 14): `K = 30`, `KSHR = 5.0` | nilai kode |
| +20,1%; +18,4%; +18,7% (2 sampai 3 meter) | C3 | `resid_knn.json`, `postproc` → `s1` → `M23`: 0,2010; 0,1837; 0,1873 | dikali 100, dibulatkan |
| +6,4%; +3,2%; +7,1% (4 meter) | C3 | `resid_knn.json`, `postproc` → `s1` → `M4`: 0,0637; 0,0318; 0,0714 | dikali 100, dibulatkan |
| k 0,505 dan 0,657; a0 0,0081 dan 0,0084; b0 0,393 dan 0,387 | C3 | `resid_knn.json`, `postproc` → `s3` (gas, EUI) | dibulatkan |
| 224 dan 104 bobot | C1 | 28 dan 13 anggota gabungan (keluaran sel 23) × 8 kolom basis (`blend.vc_basis`, spesifikasi `A2xM2xp2`) | perkalian |
