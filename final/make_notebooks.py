"""Build the two self-contained final notebooks from final/src and final/configs (the code run by run_all.py).

  final/DICEFEST2026_<TEAM>.ipynb        narrative notebook: data, domain features, multi-output architecture,
                                         training, assembly of the submission, evaluation, interpretability and
                                         policy recommendations
  final/inference_from_artifacts.ipynb   rebuilds the submission from the saved fold models in final/artifacts

Every source file is embedded verbatim as a %%writefile cell, so each notebook alone is enough to reproduce.
usage: python final/make_notebooks.py [TEAM_NAME]
"""
import os
import sys

import nbformat as nbf
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEAM_NAME", "ISI_NAMA_TIM")
SRC = ["common.py", "te.py", "features_knn.py", "fb07b.py", "train_lib.py", "nn.py", "nca.py", "blend.py",
       "adopt_v2.py", "dup_fix.py", "postprocess.py", "pipeline.py", "inference.py", "metric.py", "seed.py",
       "explain.py", "report.py"]
CONF = ["members.yaml", "resid_knn.json"]
CFG = yaml.safe_load(open(os.path.join(HERE, "configs", "members.yaml")))
N_GAS, N_EL = len(CFG["blend"]["gas"]), len(CFG["blend"]["elec"])

SETUP = '''import os, sys, platform, hashlib, json, subprocess, time
# ---- paths and switches ----
DATA_DIR = os.environ.get("DICEFEST_DATA_DIR", "data/raw")   # train.csv, test.csv, sample_submission.csv
ROOT = os.path.abspath("dicefest_nb")                         # embedded source code + work files
WORK = os.environ.get("DICEFEST_WORK", os.path.join(ROOT, "work"))
OUT_CSV = os.path.join(ROOT, "submission.csv")
TRAIN = os.environ.get("DICEFEST_TRAIN", "1") == "1"   # 0 = skip training and use member predictions already in WORK
os.environ["DICEFEST_DATA_DIR"] = os.path.abspath(DATA_DIR)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS, before CUDA starts
os.makedirs(os.path.join(ROOT, "src"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
for f in ("train.csv", "test.csv", "sample_submission.csv"):
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"missing {f} in {DATA_DIR}"
import numpy, pandas, sklearn, scipy, lightgbm, xgboost, torch, yaml, pyarrow
print("python", platform.python_version())
for m in (numpy, pandas, pyarrow, scipy, sklearn, lightgbm, xgboost, torch):
    print(f"{m.__name__:12s} {m.__version__}")
print("CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EXPECTED = {"numpy": "2.5.4", "pandas": "3.0.6", "scikit-learn": "1.9.1", "lightgbm": "4.7.0",
            "xgboost": "3.4.1", "torch": "2.14.1+cu130"}
print("versions used for the reference run:", EXPECTED)
print("TRAIN =", TRAIN, "| WORK =", WORK)'''

MODULES = """| Modul | Isi |
|---|---|
| `common.py` | pemuatan data, metrik, fitur dasar dan rasio fisik |
| `te.py`, `features_knn.py`, `fb07b.py` | encoding spasial (subzona, zona, sel iklim, jendela sub_n) dan fitur kNN-target, semuanya out-of-fold |
| `train_lib.py`, `nn.py`, `nca.py` | pelatihan anggota: GBM (XGBoost, LightGBM), MLP (PLR, TabM-mini), regresor retrieval NCA |
| `blend.py`, `postprocess.py` | blend NNLS dengan bobot bervariasi dan koreksi residual grup + kNN |
| `adopt_v2.py`, `dup_fix.py` | model adopsi pemanas listrik dan salinan baris duplikat |
| `pipeline.py` | orkestrasi: tahap anggota, adopsi, perakitan submisi |
| `explain.py`, `report.py` | interpretabilitas (TreeSHAP), indeks kebijakan per zona, tabel dan diagram notebook |
| `inference.py`, `metric.py`, `seed.py` | prediksi dari artefak, metrik kompetisi, penguncian seed |
| `configs/members.yaml`, `configs/resid_knn.json` | daftar anggota dan hiperparameternya; bobot koreksi dan parameter post-processing yang dibekukan |"""


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def hidden(cell):
    """collapse the source of a long code cell (JupyterLab / Notebook 7 / VS Code honour this metadata)"""
    cell.metadata["jupyter"] = {"source_hidden": True, "outputs_hidden": True}
    cell.metadata["tags"] = ["hide-input"]
    return cell


def embed_cells():
    cells = [md('## Modul kode (diciutkan)\nNotebook ini harus bisa berjalan sendiri, jadi seluruh kode solusi ikut di dalamnya: sel-sel di bawah menulis 17 modul Python dan 2 file konfigurasi ke folder `dicefest_nb/`, lalu meng-import-nya. Supaya alur cerita mudah dibaca, isi sel-sel ini sengaja **diciutkan**; klik tanda di sisi kiri sel untuk membukanya. Bagian-bagian berikutnya hanya memanggil fungsi dari modul ini, sehingga setiap sel analisis tetap pendek.\n\n' + MODULES)]
    for f in SRC:
        cells.append(hidden(code(f"%%writefile dicefest_nb/src/{f}\n" + open(os.path.join(HERE, "src", f)).read())))
    for f in CONF:
        cells.append(hidden(code(f"%%writefile dicefest_nb/configs/{f}\n" + open(os.path.join(HERE, "configs", f)).read())))
    cells.append(code('sys.path.insert(0, os.path.join(ROOT, "src"))\n'
                      'import pipeline, report, explain\n'
                      'from common import load, TARGET_COLS, SIGMAS\n'
                      'import numpy as np, pandas as pd, matplotlib.pyplot as plt\n'
                      'pd.set_option("display.width", 160)\n'
                      'cfg = pipeline.load_config()\n'
                      'print(len(cfg["members"]), "anggota terdefinisi;", len(cfg["blend"]["gas"]), "di blend gas/EUI,",\n'
                      '      len(cfg["blend"]["elec"]), "di blend listrik")'))
    return cells


INTRO = f"""# DICE FEST 2026: Trilema Transisi Energi Perkotaan
### Tim {TEAM}

Notebook ini berisi seluruh solusi kami, dari data mentah sampai file submisi, lalu dari model sampai rekomendasi
kebijakan. Semua angka tebakan dihasilkan oleh kode di notebook ini, dengan seed terkunci dan tanpa data dari luar.

**Gambaran singkat.** Kami memperkirakan lima angka energi untuk setiap kawasan permukiman. Sebelum melatih model, kami
membedah data dan menemukan bahwa porsi bangunan bersih bisa dihitung tepat, adopsi pemanas listrik sudah diketahui
untuk 92% kawasan uji, dan gas total serta intensitas gas (EUI) terikat rumus luas lantai. Bagian yang benar-benar
harus ditebak kami kerjakan dengan fitur yang meniru cara energi dipakai, 41 model yang digabung dengan bobot yang
menyesuaikan, koreksi galat wilayah, dan penyesuaian akhir untuk batas data. Skor leaderboard publik: 0,24913.

**Hasil utama.** Skor LB publik 0,24913 dari 41 model yang digabung. Pendorong konsumsi terbesar: ukuran rumah (arah berlawanan untuk gas dan listrik), stok bangunan lama, dan iklim; adopsi pemanas listrik 81% ditentukan infrastruktur pemanas yang ada. Peta trilema: 118 zona yang memuat 47% konsumsi gas perlu perbaikan bangunan sebelum elektrifikasi; potensi hemat 7,4 TWh; elektrifikasi pemanas penuh menaikkan beban listrik zona rata-rata 95%.

| Bagian | Isi |
|---|---|
| 1. Data | rumus tersembunyi di data dan cara data uji dipilih |
| 2. Fitur | ukuran rumah, rata-rata wilayah, kawasan paling mirip, gambaran kawasan |
| 3. Arsitektur | tiga jalur untuk lima target, 41 model, cara menggabungkan |
| 4-5. Pelatihan dan submisi | melatih semua model lalu menyusun file submisi |
| 6. Evaluasi | skor dan perjalanannya |
| 7. Kebijakan | model penjelas, SHAP, peta trilema 330 zona, rekomendasi |
| Modul kode | seluruh kode solusi, ditulis ke disk oleh notebook ini (diciutkan, letaknya setelah bagian 0) |

**Cara menjalankan.** Pelatihan penuh butuh GPU 16 GB, 16 vCPU, RAM 32 GB, sekitar 20 jam (tiga jalur paralel).
Kalau tebakan semua model sudah ada di folder `WORK`, set `DICEFEST_TRAIN=0` untuk melewati pelatihan; semua bagian lain
tetap dijalankan. File submisi yang dihasilkan selalu dicetak sidik sha256-nya, dan pada run kami hasilnya
`aec745ab...d66e`, sama persis dengan file yang kami kirim."""


TRAIN_NOTE = '\n\n*Catatan untuk salinan notebook ini:* sel pelatihan di bawah dilewati (`DICEFEST_TRAIN=0`) karena butuh sekitar 20 jam. Kami menjalankan pelatihannya terpisah dengan kode yang sama persis (modul di lampiran): ke-41 model dilatih ulang dari nol dan setiap tebakannya identik sampai byte terakhir dengan versi pengembangan. Sel-sel berikutnya memakai tebakan dari run itu, dan file submisi yang dihasilkan (sha256 di bagian 5) sama persis dengan file yang kami kirim. Jalankan dengan `DICEFEST_TRAIN=1` untuk melatih ulang semuanya.'


def full_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [md(INTRO), md("## 0. Persiapan\nLokasi data, folder kerja, dan versi pustaka yang dipakai."), code(SETUP)] + embed_cells() + [
        md("""## 1. Data: apa yang kami temukan sebelum memodelkan
Satu baris adalah satu kawasan kecil (kira-kira satu kode pos, rata-rata 22 meter). Sel di bawah memeriksa empat hal
yang menentukan seluruh desain solusi:

* **EUI x luas lantai = gas.** Kolom `bld_dim_06` ternyata luas lantai, dan EUI hanyalah gas dibagi luas itu. Jadi
  gas dan EUI dilayani satu keluarga model.
* **Porsi bangunan bersih = 100 x `bld_dim_02` / `bld_dim_01`**, dibulatkan dua desimal. Target ini dihitung, tidak
  ditebak.
* **Adopsi pemanas listrik sama untuk satu subzona**, dan 92% kawasan uji ada di subzona yang muncul di data latih.
* **Data uji diambil acak per baris** (tepat 30%). Karena itu kami menguji model dengan membagi data latih secara acak
  menjadi 10 bagian, dan rata-rata wilayah boleh dipakai sebagai fitur asalkan dihitung tanpa melihat bagian yang
  sedang diuji."""),
        code('tr, te = load()\nprint("train", tr.shape, "| test", te.shape)\nreport.identity_checks(tr, te)'),
        md("Sebaran target. Gas, listrik, dan EUI digambar dalam skala logaritma karena rentangnya sangat lebar."),
        code('''fig, axs = plt.subplots(1, 4, figsize=(15, 3))
for ax, c in zip(axs, TARGET_COLS[:3] + [TARGET_COLS[4]]):
    v = tr[c].values
    ax.hist(np.log10(v) if c != TARGET_COLS[4] else v, bins=80, color="#3b6ea5")
    ax.set_title(c + (" (log10)" if c != TARGET_COLS[4] else ""), fontsize=9)
plt.tight_layout(); plt.show()
print("galat tiap target dibagi sigma berikut sebelum dirata-rata:", SIGMAS)'''),
        md("""## 2. Fitur: menerjemahkan fisika dan geografi ke dalam angka
Pertanyaan yang memandu kami: apa yang membuat satu kawasan memakai energi lebih banyak dari kawasan lain?

**Ukuran rumah, bukan ukuran kawasan.** Gas total terutama mengikuti berapa banyak rumah di kawasan. Yang lebih
informatif adalah perilaku per rumah, jadi kami membuat rasio seperti luas lantai per meter (ukuran rata-rata rumah),
luas per sertifikat energi, dan sisa porsi jenis pemanas (sebagian besar pemanas listrik).

**Menebak per rumah.** Model menebak gas per meter, per meter persegi, per sertifikat, atau logaritmanya, lalu
hasilnya dikalikan kembali. Bobot tiap baris diatur supaya kesalahan yang dihukum tetap kesalahan pada skala asli.

**Rata-rata wilayah.** Kawasan yang berdekatan cenderung mirip. Kami memakai rata-rata target di subzona, zona, petak
iklim (gabungan tiga kolom iklim yang membentuk grid berisi sekitar 15 kawasan), dan kawasan di sekitar nomor subzona
yang berurutan secara geografis. Semuanya dihitung dari bagian data yang tidak sedang diuji, supaya model tidak
mencontek.

**Kawasan paling mirip.** Untuk setiap kawasan, kami mencari kawasan di sekitarnya yang stok bangunannya paling mirip
(ukuran rumah, cakupan sertifikat, campuran era bangunan) dan memakai rata-rata konsumsi mereka. Ini ide dengan efek
terbesar di proyek kami: galat EUI satu model turun 0,0022.

**Gambaran kawasan.** Jumlah kawasan, total luas, meter, dan sertifikat per subzona dan zona dari data latih dan uji
(tanpa target).

**Jumlah rumah tangga untuk adopsi.** Kolom campuran pemanas adalah pecahan cacah rumah tangga. Bilangan bulat terkecil
yang membuat semua pecahan utuh adalah jumlah rumah tangga, sehingga adopsi subzona baru bisa dimodelkan sebagai
peluang "berapa rumah dari K rumah memakai listrik".

Sel di bawah membangun fitur dasar dan gambaran kawasan, lalu menunjukkan kelompok fiturnya dan beberapa korelasi."""),
        code('''from train_lib import Data
d = Data()
ex, ext = d.agg_features()
fam = pd.Series([explain.family(c) for c in list(d.X.columns) + list(ex.columns)]).value_counts()
print("jumlah fitur per kelompok:"); print(fam.to_string())
demo = pd.DataFrame({"log EUI": np.log(tr[TARGET_COLS[2]]), "log luas per meter": np.log(d.X["area_per_meter"]),
                     "sisa porsi pemanas": d.X["hvac_rest"], "adopsi": tr[TARGET_COLS[4]]})
print("\\nkorelasi beberapa fitur turunan dengan target:"); print(demo.corr().round(3).to_string())'''),
        md("""## 3. Arsitektur: tiga jalur untuk lima target
* **Gas dan EUI** dilayani satu keluarga model. Saat menggabungkan model, bobot untuk kolom gas menekankan kawasan
  besar (yang paling menentukan galat gas), sedangkan bobot untuk kolom EUI tidak.
* **Listrik** punya keluarga model sendiri.
* **Porsi bangunan bersih** dihitung dengan rumus; **adopsi** disalin per subzona dan sisanya ditebak dengan model
  peluang.

Model yang dipakai sengaja beragam karena tiap jenis melihat data dengan cara berbeda: XGBoost dan LightGBM (pohon
keputusan bertingkat), beberapa jaringan saraf, dan NCA, jaringan yang belajar sendiri kawasan mana yang mirip lalu
menebak dari rata-rata kawasan mirip itu. Gabungannya memakai bobot tidak negatif yang berubah perlahan mengikuti luas
kawasan, jumlah meter, dan besar tebakan, karena model yang unggul di kawasan kecil belum tentu unggul di kawasan besar.

Setelah digabung, tebakan setiap kawasan digeser sedikit mengikuti rata-rata galat tetangga yang mirip (galat yang
masih berkorelasi antar kawasan berdekatan). Terakhir ada penyesuaian untuk dua batas di data: kawasan dengan 2 sampai
4 meter yang selalu ditebak terlalu rendah, dan gas yang di data latih tidak pernah melewati satu juta kWh. Semua
besaran untuk langkah setelah penggabungan ditentukan pada 30% data latih yang disisihkan sebagai ujian internal."""),
        code('report.architecture_diagram()\nreport.member_table(cfg)'),
        md("""## 4. Melatih semua model
Tiga proses berjalan paralel: jalur `xgb` (XGBoost di GPU), `nn` (jaringan saraf dan NCA di GPU), dan `cpu` (LightGBM).
Setiap model dilatih 10 kali (10 bagian data, seed 42) dan menyimpan tebakan untuk data latih (dari bagian yang tidak
ia lihat) dan data uji (rata-rata 10 model). Semua pustaka dipaksa deterministik, sehingga run ulang di hardware yang
sama memberi angka yang identik sampai byte terakhir.""" + TRAIN_NOTE),
        code('''import shutil
if TRAIN:
    if os.path.exists(WORK):
        shutil.rmtree(WORK)          # mulai dari kondisi bersih
    os.makedirs(WORK)
    t0 = time.time()
    procs = {}
    for lane in pipeline.LANES:   # XGBoost (GPU), jaringan saraf + NCA (GPU), LightGBM (CPU) paralel
        env = dict(os.environ, OMP_NUM_THREADS=pipeline.LANE_THREADS[lane])
        cmd = [sys.executable, "-u", "-c",
               f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); import pipeline; "
               f"pipeline.run_members({lane!r}, {WORK!r})"]
        procs[lane] = subprocess.Popen(cmd, stdout=open(os.path.join(WORK, f"lane_{lane}.log"), "w"),
                                       stderr=subprocess.STDOUT, env=env)
    for lane, p in procs.items():
        rc = p.wait()
        print(f"jalur {lane} selesai rc={rc} setelah {(time.time() - t0) / 3600:.2f} jam")
        assert rc == 0, open(os.path.join(WORK, f"lane_{lane}.log")).read()[-3000:]
need = sorted(set(cfg["blend"]["gas"] + cfg["blend"]["elec"]))
missing = [m for m in need if not os.path.exists(os.path.join(WORK, "pred", m + ".npy"))]
print(len(need), "model di gabungan;", "semua tebakan tersedia" if not missing else f"belum ada: {missing}")'''),
        md("## 5. Menyusun file submisi\nModel adopsi, penggabungan, koreksi galat wilayah, penyesuaian akhir, salinan 77 baris duplikat, lalu file submisi beserta sidik sha256-nya."),
        code('info = pipeline.run_adoption(WORK)\nprint(info)\nsha = pipeline.assemble(WORK, OUT_CSV)\n'
             'summary = json.load(open(os.path.join(WORK, "summary.json")))\nprint("sha256", sha)'),
        md("""## 6. Evaluasi
Skor jujur hasil penggabungan per kolom (bobot dihitung di empat per lima data, diuji di sisanya), lalu perjalanan skor
leaderboard. Setiap langkah dipilih dari ujian internal, dan leaderboard selalu membaik hampir persis sebesar
perkiraannya."""),
        code('''cv = summary["cv_blend_honest"]
print({k: round(v, 5) for k, v in cv.items()}, "(sebelum koreksi wilayah dan penyesuaian akhir)")
hist = pd.DataFrame([("v12", "gambaran kawasan + jaringan saraf PLR", 0.25025), ("v15", "koreksi galat wilayah", 0.24954),
                     ("v16", "fitur kawasan paling mirip di dalam model", 0.24943),
                     ("v19", "objektif Poisson + kawasan mirip lanjutan", 0.24934),
                     ("v19vc", "bobot gabungan yang menyesuaikan", 0.24925), ("v19nca", "model NCA", 0.24920),
                     ("final", "penyesuaian meter sedikit + batas gas", 0.24913)],
                    columns=["versi", "langkah", "LB publik"])
hist'''),
        code('sub = pd.read_csv(OUT_CSV)\nprint(sub.shape)\nsub.describe().T.round(3)'),
        md("""## 7. Apa artinya bagi kebijakan
Gabungan 41 model terlalu rumit untuk dibaca, dan sebagian fiturnya (rata-rata target wilayah) bukan sesuatu yang
bisa diubah pembuat kebijakan. Karena itu kami melatih satu **model penjelas** per target yang hanya memakai
karakteristik kawasan yang bisa diukur: bangunan, era pembangunan, campuran pemanas, sosio-demografi, iklim, dan
gambaran kawasan.

Untuk membacanya kami memakai **SHAP**. Bayangkan tebakan model sebagai tumpukan sumbangan: dimulai dari rata-rata
nasional, lalu setiap fitur menambah atau mengurangi. SHAP menghitung besar sumbangan setiap fitur untuk setiap
kawasan secara adil. Kesetiaan model penjelas pada data ditunjukkan oleh R² pada data yang tidak ia lihat.

Nama kolom disamarkan panitia; makna yang kami tulis (misalnya `hvac_dist_02` sebagai porsi pemanas gas jaringan)
adalah dugaan dari pola datanya."""),
        code('''X, Xt = explain.feature_frame(d, ex, ext)
models, shap_res, summ = {}, {}, {}
for t, (label, fn) in explain.TARGETS.items():
    y = fn(tr)
    models[t], r2 = explain.fit_explain(X, y)
    shap_res[t] = explain.shap_values(models[t], X)
    summ[t] = dict(label=label, r2_holdout=round(r2, 3),
                   families=explain.family_shares(shap_res[t][1], list(X.columns)).round(4).to_dict())
pd.DataFrame({t: {"target": s["label"], "R2 pada data tak terlihat": s["r2_holdout"]} for t, s in summ.items()}).T'''),
        md("Porsi sumbangan tiap kelompok fitur. Konsumsi gas dan listrik digerakkan oleh rasio fisik dan sosio-demografi; adopsi hampir seluruhnya oleh campuran jenis pemanas yang sudah ada."),
        code('explain.plot_families(summ)'),
        md("Sepuluh fitur dengan sumbangan terbesar untuk tiga target utama."),
        code('for t in ("eui", "epm", "adopt"):\n    print(explain.TARGETS[t][0]); display(explain.top_features(shap_res[t][1], list(X.columns), k=10))'),
        md("Bagaimana sumbangan berubah mengikuti nilai fitur. Setiap titik adalah satu kawasan, garis merah adalah rata-ratanya."),
        code('''explain.plot_dependence(X, shap_res, [
    ("eui", "area_per_meter", "luas lantai per meter (m2)", True),
    ("epm", "area_per_meter", "luas lantai per meter (m2)", True),
    ("adopt", "hvac_dist_02", "porsi pemanas gas jaringan (%)", False),
    ("eui", "env_climate_05", "env_climate_05 (iklim, anonim)", False),
    ("eui", "bld_era_15", "porsi bangunan periode bld_era_15 (%)", False),
    ("eui", "bld_era_09", "porsi bangunan periode bld_era_09 (%)", False)])'''),
        md("""### 7.1 Membaca grafik di atas
* **Ukuran rumah bekerja ke arah berlawanan untuk gas dan listrik.** Rumah kecil memakai gas per meter persegi sekitar
  65% lebih banyak daripada rumah besar, karena dinding dan atapnya relatif luas dibanding ruang yang dipanaskan.
  Sebaliknya, rumah besar memakai listrik per rumah sekitar 75% lebih banyak.
* **Bangunan dari periode tertentu lebih boros** (hingga sekitar 7% EUI), dan **iklim** menggeser EUI sekitar 15%.
  Inilah bagian konsumsi yang bisa dipangkas lewat perbaikan dinding, atap, dan jendela.
* **Adopsi pemanas listrik ditentukan infrastruktur yang ada.** Porsi pemanas gas jaringan yang tinggi menekan adopsi;
  faktor sosio-demografi hanya menyumbang sekitar 15%.

### 7.2 Peta trilema per zona
Untuk menjadikan temuan ini daftar kerja, kami menggabungkan nilai sebenarnya (kawasan latih) dan tebakan akhir
(kawasan uji), lalu meringkasnya ke 330 zona dengan dua ukuran: **potensi hemat dari perbaikan bangunan** (gas yang
hilang bila sumbangan stok bangunan di atas rata-rata nasional dihapus) dan **kenaikan beban listrik bila pemanas gas
diganti pompa panas listrik** (asumsi: 80% gas untuk pemanasan, 1 kWh listrik menghasilkan 3 kWh panas)."""),
        code('''sub_best = pd.read_csv(OUT_CSV)
Z = explain.zone_indices(models["eui"], X, Xt, tr, te, sub_best)
q = explain.quadrant_table(Z)
explain.plot_trilemma(Z)
q.round(3)'''),
        code('''tot = dict(gas_TWh=Z.gas.sum() / 1e9, potensi_hemat_bangunan_TWh=Z.gas_saving.sum() / 1e9,
           potensi_hemat_pct=100 * Z.gas_saving.sum() / Z.gas.sum(),
           kenaikan_beban_median_pct=Z.load_increase_pct.median())
print({k: round(v, 2) for k, v in tot.items()})
cols_show = ["gas", "eui", "adopt", "gas_network_share", "saving_pct", "load_increase_pct", "kuadran"]
print("10 zona dengan potensi hemat bangunan terbesar (kWh):")
display(Z.sort_values("gas_saving", ascending=False).head(10)[cols_show].round(2))
print("10 zona dengan kenaikan beban listrik terbesar bila pemanas dielektrifikasi:")
display(Z.sort_values("load_increase_pct", ascending=False).head(10)[cols_show].round(2))'''),
        md("""### 7.3 Rekomendasi
1. **Urutkan program per kelompok zona, jangan seragamkan.** Kelompok Q1 (boros dan jaringan rawan) memuat hampir
   separuh konsumsi gas dan separuh potensi hemat bangunan. Di sini perbaikan bangunan harus datang sebelum pompa
   panas: kebutuhan panas yang turun lebih dulu memangkas tagihan dan emisi, sekaligus mengecilkan lonjakan beban
   listrik saat elektrifikasi masuk.
2. **Jadikan Q2 kemenangan cepat untuk pompa panas.** Intensitas gasnya tinggi tetapi jaringan masih punya ruang.
3. **Ajak operator jaringan sejak awal di Q1 dan Q3.** Zona dengan kenaikan beban lebih dari dua kali lipat butuh
   penguatan gardu, tarif berdasarkan waktu pemakaian, atau pompa panas dengan penyimpan panas sebelum subsidi besar.
4. **Beri insentif khusus untuk kawasan yang sudah tersambung gas**, karena tanpa dorongan adopsi di sana nyaris nol.
5. **Pisahkan target efisiensi dan target emisi.** Rumah kecil boros per meter persegi, rumah besar boros per rumah.

**Batasan.** SHAP menunjukkan hubungan dalam data, bukan sebab-akibat yang pasti. Skenario beban listrik adalah
perhitungan fisika sederhana di atas tebakan model; urutan antar zona lebih bisa dipegang daripada angka mutlaknya."""),
        md("""## 8. Penutup
* Memahami data lebih dulu menyelesaikan dua target tanpa model dan menentukan arah seluruh pekerjaan.
* Perbaikan terbesar datang dari informasi tetangga yang dihitung dengan jujur: rata-rata wilayah, kawasan paling mirip,
  dan koreksi galat wilayah. Sisanya dari keragaman model dan penggabungan yang menyesuaikan.
* Setiap perubahan dipilih dari ujian internal yang terbukti memperkirakan leaderboard, bukan dari leaderboard itu sendiri.
* Model yang sama bisa diubah menjadi peta prioritas yang langsung bisa dipakai: zona mana yang perlu dibenahi
  bangunannya dulu, mana yang siap beralih ke listrik, dan mana yang jaringannya harus diperkuat lebih dulu."""),
    ]
    return nb


def inference_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# DICE FEST 2026: submisi dari artefak model tersimpan\n\n"
           "Memuat model per fold dari `final/artifacts/models/` (checksum di `final/artifacts/SHA256SUMS`), "
           "membuat ulang prediksi OOF dan test setiap anggota tanpa training, lalu menjalankan tahap adopsi, "
           "blend, koreksi, dan pasca-proses yang sama. Hasilnya harus identik byte dengan output notebook utama."),
        code(SETUP + '\nARTIFACTS = os.environ.get("DICEFEST_ARTIFACTS", "final/artifacts")'),
    ] + embed_cells() + [
        md("## Verifikasi checksum artefak"),
        code('''bad = 0
for line in open(os.path.join(ARTIFACTS, "SHA256SUMS")):
    h, rel = line.split(maxsplit=1)
    p = os.path.join(ARTIFACTS, rel.strip())
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
        bad += 1
        print("checksum mismatch", rel)
assert bad == 0
print("all artifact checksums ok")'''),
        md("## Prediksi dari model tersimpan"),
        code('import inference\nos.makedirs(WORK, exist_ok=True)\n'
             'inference.predict_all(os.path.join(ARTIFACTS, "models"), WORK)'),
        md("## Adopsi, blend, koreksi, submisi"),
        code('info = pipeline.run_adoption(WORK)\nsha = pipeline.assemble(WORK, OUT_CSV)\nprint("sha256", sha)'),
    ]
    return nb


if __name__ == "__main__":
    nbf.write(full_notebook(), os.path.join(HERE, f"DICEFEST2026_{TEAM}.ipynb"))
    nbf.write(inference_notebook(), os.path.join(HERE, "inference_from_artifacts.ipynb"))
    print("notebooks written for team", TEAM)
