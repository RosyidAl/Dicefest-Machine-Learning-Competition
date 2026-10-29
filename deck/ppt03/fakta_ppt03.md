# Fakta Tambahan untuk *Deck* v4 (ppt03)

Lembar fakta ini melengkapi kelima lembar fakta *deck* v3 (`../fakta_notebook.txt`, `../fakta_turunan.md`,
`../ppt01/fakta_tambahan.md`, *writeup* B, dan `../ppt02/fakta_ppt02.md`). Isinya hanya angka di
`01_struktur_deck_v4.md` yang belum tercetak di kelima berkas tersebut. `cek_struktur.py` memakai keenam berkas.

| Angka | Dipakai di | Asal | Keterangan |
|---|---|---|---|
| REG_273: peringkat kenaikan relatif 47 (COP 3 seragam) dan 32 (COP per iklim) | C10 | `../ppt02/bukti/cek_zona_contoh.json`, kunci `peringkat_kenaikan_dasar` dan `peringkat_kenaikan_cop_iklim` | REG_273 titik awal R1 karena tambahan absolutnya peringkat 2, bukan karena kenaikan relatifnya |
| 25%, 30%, dan 26% (porsi SHAP rasio fisik turunan) | *slide* 3 | `make_visuals_v4.py`: `explain.family_shares` pada `reports/policy/shap_eui.npz`, `shap_gpm.npz`, `shap_epm.npz` | sama dengan *writeup* B Tabel 3 (diperiksa dengan *assertion*) |
| 0,25025; 0,24954; 0,24943; 0,24934; 0,24925; 0,24920; 0,24913 | *slide* 7 | keluaran sel 37, diparse oleh `make_visuals_v4.py` | skor *leaderboard* publik per langkah |
| 74,6% (dibulatkan +75%) | *slide* 10 | `../ppt01/bukti/cek_kebijakan.json`, `kontras_ukuran_listrik_per_rumah` | sel 48 butir 1 menulis "sekitar 75%" |
