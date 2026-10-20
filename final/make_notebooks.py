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


def embed_cells():
    cells = [md("## Lampiran kode: modul pipeline\nSel-sel berikut menulis modul ke `dicefest_nb/src/` dan konfigurasi ke "
                "`dicefest_nb/configs/`, lalu meng-import-nya. Isinya identik dengan `final/src/` dan `final/configs/`. "
                "Alur analisis di bagian selanjutnya hanya memanggil fungsi dari modul ini, sehingga setiap sel tetap "
                "pendek dan mudah dibaca.\n\n" + MODULES)]
    for f in SRC:
        cells.append(code(f"%%writefile dicefest_nb/src/{f}\n" + open(os.path.join(HERE, "src", f)).read()))
    for f in CONF:
        cells.append(code(f"%%writefile dicefest_nb/configs/{f}\n" + open(os.path.join(HERE, "configs", f)).read()))
    cells.append(code('sys.path.insert(0, os.path.join(ROOT, "src"))\n'
                      'import pipeline, report, explain\n'
                      'from common import load, TARGET_COLS, SIGMAS\n'
                      'import numpy as np, pandas as pd, matplotlib.pyplot as plt\n'
                      'pd.set_option("display.width", 160)\n'
                      'cfg = pipeline.load_config()\n'
                      'print(len(cfg["members"]), "anggota terdefinisi;", len(cfg["blend"]["gas"]), "di blend gas/EUI,",\n'
                      '      len(cfg["blend"]["elec"]), "di blend listrik")'))
    return cells


INTRO = f"""# DICE FEST 2026: The Urban Energy Transition Trilemma
### Tim {TEAM}

Notebook ini berisi solusi lengkap kami: dari data mentah sampai file submisi, lalu interpretasi model untuk
rekomendasi kebijakan. Semua angka prediksi keluar dari kode di notebook ini, dengan seed terkunci dan tanpa data
eksternal.

**Ringkasan solusi**

| Komponen | Isi |
|---|---|
| Validasi | KFold acak 10 lipat (test diambil acak per baris, 70/30) + holdout internal 70/30 untuk bobot pasca-blend |
| Identitas eksak | gas = EUI x luas lantai; rasio bangunan bersih = bld_dim_02 / bld_dim_01; adopsi konstan per subzona |
| Fitur domain | rasio fisik (luas per meter, per sertifikat), agregat kawasan, encoding spasial out-of-fold, kNN-target di ruang stok bangunan |
| Model | {N_GAS} anggota untuk keluarga gas/EUI dan {N_EL} untuk listrik: XGBoost, LightGBM (termasuk Poisson), MLP (PLR, TabM-mini, multi-task), regresor retrieval NCA |
| Ensemble | blend NNLS dengan bobot yang bervariasi menurut luas, jumlah meter, dan level prediksi; koreksi residual spasial |
| Pasca-proses | koreksi bias baris bermeter sangat sedikit, batas truncation gas 1e6 kWh, salinan 77 baris duplikat |
| Skor | LB publik 0,24913 (metrik: rata-rata RMSE/sigma lima target) |

**Isi**: 0 Setup, 1 Data dan identitas fisik, 2 Rekayasa fitur domain, 3 Arsitektur multi-output, 4 Pelatihan,
5 Perakitan submisi, 6 Evaluasi, 7 Interpretabilitas dan rekomendasi kebijakan, 8 Kesimpulan, Lampiran kode.

**Menjalankan**: hardware referensi NVIDIA RTX A4000 16 GB, 16 vCPU, RAM 32 GB; pelatihan penuh sekitar 20 jam
(tiga jalur paralel). Set `DICEFEST_TRAIN=0` untuk melewati pelatihan bila prediksi anggota sudah ada di `WORK`
(semua bagian lain tetap dijalankan). Untuk verifikasi cepat dari model tersimpan tersedia
`inference_from_artifacts.ipynb`."""


def full_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [md(INTRO), md("## 0. Setup"), code(SETUP)] + embed_cells() + [
        md("""## 1. Data dan identitas fisik
Sebelum memodelkan, kami memeriksa struktur data. Tiga temuan membentuk seluruh desain:

1. **EUI adalah gas dibagi luas lantai** (`bld_dim_06`) dengan presisi mesin. Jadi gas total dan EUI dimodelkan oleh
   satu keluarga model; hanya bobot blend yang di-fit terpisah karena metriknya menekankan baris yang berbeda (gas oleh
   kawasan besar, EUI oleh kawasan dengan luas kecil).
2. **Rasio bangunan bersih adalah identitas eksak** dari dua kolom bangunan, sehingga tidak perlu model sama sekali.
3. **Adopsi pemanas listrik konstan dalam satu subzona** dan 92% baris test berada di subzona yang ada di train.
   Untuk sisanya, adopsi bisa diturunkan dari jumlah rumah tangga yang direkonstruksi dari pangsa jenis pemanas.

Test diambil acak per baris (persis 30%), sehingga KFold acak per baris meniru kondisi test dan informasi tingkat
wilayah (target encoding subzona, zona, sel iklim) sah dipakai asalkan dihitung out-of-fold."""),
        code('tr, te = load()\nprint("train", tr.shape, "| test", te.shape)\nreport.identity_checks(tr, te)'),
        code('''fig, axs = plt.subplots(1, 4, figsize=(15, 3))
for ax, c in zip(axs, TARGET_COLS[:3] + [TARGET_COLS[4]]):
    v = tr[c].values
    ax.hist(np.log10(v) if c != TARGET_COLS[4] else v, bins=80, color="#3b6ea5")
    ax.set_title(c + (" (log10)" if c != TARGET_COLS[4] else ""), fontsize=9)
plt.tight_layout(); plt.show()
print("sigma beku metrik:", SIGMAS)'''),
        md("""## 2. Rekayasa fitur domain
Fitur kami dibangun dari logika fisik dan spasial konsumsi energi kawasan, bukan dari pencarian buta.

**2.1 Rasio fisik.** Konsumsi energi kawasan adalah jumlah dari banyak rumah tangga. Karena itu kami menormalkan ukuran:
`area_per_meter` (luas lantai per meter = ukuran hunian rata-rata), `area_per_epc` (luas per sertifikat energi),
`meters_per_epc` (cakupan sertifikat), `clean_ratio`, dan `hvac_rest` (100 dikurangi semua pangsa jenis pemanas, yaitu
pangsa pemanas listrik + kategori yang hilang).

**2.2 Target dalam bentuk intensitas.** Anggota tidak memprediksi gas total secara langsung, tetapi gas per meter,
gas per luas, gas per sertifikat, atau log-nya, dengan bobot baris yang menyamakan loss dengan galat skala asli.
Variansi residual mengikuti rata-rata rumah tangga (`a + b/M`, M = jumlah meter), sehingga bobot `M` atau `1/M`
dipakai sesuai formulasi. Beberapa anggota memakai objektif Poisson untuk laju per meter.

**2.3 Encoding spasial out-of-fold.** Rata-rata target (dalam bentuk intensitas) di subzona, zona, sel grid iklim
(gabungan `env_climate_01/03/05`) dan jendela tetangga sepanjang nomor subzona (`sub_n` berurutan secara spasial),
semuanya dihitung hanya dari baris fit tiap lipatan dengan cross-fitting di dalamnya.

**2.4 kNN-target di ruang stok bangunan.** Untuk tiap baris, rata-rata target dari k kawasan paling mirip
(ruang: log luas/meter, log sertifikat/meter, porsi era bangunan, porsi bersih, dan lainnya) di dalam jendela spasial.
Versi lanjutan memakai ruang era, ruang bangunan + sensus (PCA), dan ruang ukuran. Fitur ini memberi perbaikan model
tunggal terbesar (EUI -0,0022).

**2.5 Agregat kawasan transduktif.** Jumlah baris, total luas, meter, dan sertifikat per subzona dan zona dari baris
train dan test (tanpa target), serta porsi baris terhadap kawasannya.

**2.6 Rekonstruksi rumah tangga untuk adopsi.** Pangsa jenis pemanas adalah pecahan cacah/N. N terkecil yang membuat
semua pangsa menjadi bilangan bulat adalah jumlah rumah tangga; dari situ adopsi = 100 (K - r) / N dengan K = sisa
rumah tangga, sehingga adopsi subzona baru dimodelkan sebagai proporsi binomial."""),
        code('''from train_lib import Data
d = Data()
ex, ext = d.agg_features()
fam = pd.Series([explain.family(c) for c in list(d.X.columns) + list(ex.columns)]).value_counts()
print("fitur dasar + agregat per keluarga:"); print(fam.to_string())
demo = pd.DataFrame({"log EUI": np.log(tr[TARGET_COLS[2]]), "log luas per meter": np.log(d.X["area_per_meter"]),
                     "hvac_rest": d.X["hvac_rest"], "adopsi": tr[TARGET_COLS[4]]})
print("\\nkorelasi beberapa fitur turunan dengan target:"); print(demo.corr().round(3).to_string())'''),
        md("""## 3. Arsitektur multi-output
Lima kolom target ditangani oleh tiga jalur yang memanfaatkan strukturnya:

* **Keluarga gas/EUI**: satu set anggota memprediksi EUI (dalam berbagai formulasi intensitas); bobot blend untuk kolom
  gas di-fit dengan bobot baris luas lantai sehingga loss-nya sama dengan galat gas, dan untuk kolom EUI tanpa bobot.
* **Keluarga listrik**: anggota terpisah untuk listrik per meter / per luas.
* **Identitas**: rasio bersih dihitung langsung; adopsi disalin per subzona dan dimodelkan binomial untuk subzona baru.

Blend memakai NNLS dengan bobot yang berubah mulus menurut peringkat luas, jumlah meter, dan level prediksi (basis
hat multilinear), lalu koreksi residual spasial (grup + kNN) dengan bobot yang di-fit pada holdout internal agar
tidak ada label validasi yang bocor."""),
        code('report.architecture_diagram()\nreport.member_table(cfg)'),
        md("""## 4. Pelatihan semua anggota
Tiga proses paralel: jalur `xgb` (XGBoost CUDA), `nn` (MLP dan NCA di PyTorch) dan `cpu` (LightGBM, 12 thread). Setiap
anggota dilatih dengan KFold 10 lipat (seed 42) dan menyimpan prediksi OOF dan test di `WORK`. Semua library memakai
mode deterministik, sehingga run ulang di hardware yang sama memberi array yang identik byte."""),
        code('''import shutil
if TRAIN:
    if os.path.exists(WORK):
        shutil.rmtree(WORK)          # reproduce from a clean state
    os.makedirs(WORK)
    t0 = time.time()
    procs = {}
    for lane in pipeline.LANES:   # XGBoost (GPU), MLP + NCA (GPU), LightGBM (CPU) in parallel
        env = dict(os.environ, OMP_NUM_THREADS=pipeline.LANE_THREADS[lane])
        cmd = [sys.executable, "-u", "-c",
               f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); import pipeline; "
               f"pipeline.run_members({lane!r}, {WORK!r})"]
        procs[lane] = subprocess.Popen(cmd, stdout=open(os.path.join(WORK, f"lane_{lane}.log"), "w"),
                                       stderr=subprocess.STDOUT, env=env)
    for lane, p in procs.items():
        rc = p.wait()
        print(f"lane {lane} rc={rc} after {(time.time() - t0) / 3600:.2f} h")
        assert rc == 0, open(os.path.join(WORK, f"lane_{lane}.log")).read()[-3000:]
need = sorted(set(cfg["blend"]["gas"] + cfg["blend"]["elec"]))
missing = [m for m in need if not os.path.exists(os.path.join(WORK, "pred", m + ".npy"))]
print(len(need), "anggota blend;", "semua prediksi tersedia" if not missing else f"belum ada: {missing}")'''),
        md("## 5. Perakitan submisi\nModel adopsi, blend, koreksi residual, pasca-proses, salinan baris duplikat, lalu file submisi dan sha256-nya."),
        code('info = pipeline.run_adoption(WORK)\nprint(info)\nsha = pipeline.assemble(WORK, OUT_CSV)\n'
             'summary = json.load(open(os.path.join(WORK, "summary.json")))\nprint("sha256", sha)'),
        md("""## 6. Evaluasi
Skor blend jujur (bobot di-fit di empat per lima baris, dinilai di sisanya) per kolom, lalu perkembangan skor selama
kompetisi. Holdout internal memprediksi perubahan LB publik dengan baik di setiap langkah, sehingga kami memakainya
sebagai satu-satunya dasar keputusan, bukan LB publik."""),
        code('''cv = summary["cv_blend_honest"]
print({k: round(v, 5) for k, v in cv.items()}, "(sebelum koreksi residual dan pasca-proses)")
hist = pd.DataFrame([("v12", "agregat kawasan + MLP PLR", 0.25025), ("v15", "koreksi residual kNN", 0.24954),
                     ("v16", "fitur kNN-target di model", 0.24943), ("v19", "objektif Poisson + kNN-target lanjutan", 0.24934),
                     ("v19vc", "blend bobot bervariasi", 0.24925), ("v19nca", "anggota retrieval NCA", 0.24920),
                     ("final", "pasca-proses meter kecil + batas truncation gas", 0.24913)],
                    columns=["versi", "langkah", "LB publik"])
hist'''),
        code('sub = pd.read_csv(OUT_CSV)\nprint(sub.shape)\nsub.describe().T.round(3)'),
        md("""## 7. Interpretabilitas dan rekomendasi kebijakan
Ensemble submisi terlalu besar untuk dibaca langsung. Untuk analisis kebijakan kami melatih satu **model penjelas**
LightGBM per target yang hanya memakai karakteristik kawasan yang bisa diukur (bangunan, era, bauran pemanas,
sosio-demografi, iklim, agregat kawasan) tanpa encoding target. Dengan begitu setiap kontribusi SHAP merujuk pada sesuatu
yang bisa diubah atau ditargetkan oleh pembuat kebijakan. SHAP dihitung eksak (TreeSHAP bawaan LightGBM) pada 20 ribu
baris acak; kesesuaian model penjelas dilaporkan sebagai R² pada lipatan holdout.

Nama kolom dianonimkan panitia; makna yang kami tulis (misalnya pangsa pemanas gas jaringan untuk `hvac_dist_02`)
adalah dugaan dari struktur data dan kami tandai sebagai dugaan."""),
        code('''X, Xt = explain.feature_frame(d, ex, ext)
models, shap_res, summ = {}, {}, {}
for t, (label, fn) in explain.TARGETS.items():
    y = fn(tr)
    models[t], r2 = explain.fit_explain(X, y)
    shap_res[t] = explain.shap_values(models[t], X)
    summ[t] = dict(label=label, r2_holdout=round(r2, 3),
                   families=explain.family_shares(shap_res[t][1], list(X.columns)).round(4).to_dict())
pd.DataFrame({t: {"target": s["label"], "R2 holdout": s["r2_holdout"]} for t, s in summ.items()}).T'''),
        code('explain.plot_families(summ)'),
        code('for t in ("eui", "epm", "adopt"):\n    print(explain.TARGETS[t][0]); display(explain.top_features(shap_res[t][1], list(X.columns), k=10))'),
        code('''explain.plot_dependence(X, shap_res, [
    ("eui", "area_per_meter", "luas lantai per meter (m2)", True),
    ("epm", "area_per_meter", "luas lantai per meter (m2)", True),
    ("adopt", "hvac_dist_02", "pangsa pemanas gas jaringan (%)", False),
    ("eui", "env_climate_05", "env_climate_05 (iklim, anonim)", False),
    ("eui", "bld_era_15", "porsi bangunan periode bld_era_15 (%)", False),
    ("eui", "bld_era_09", "porsi bangunan periode bld_era_09 (%)", False)])'''),
        md("""### 7.1 Membaca pendorong konsumsi
* **Ukuran hunian adalah tuas fisik terbesar.** Intensitas gas per m² turun tajam seiring luas per meter, sedangkan
  listrik per meter naik. Hunian kecil boros per m² (selubung relatif besar terhadap volume), hunian besar boros per rumah.
* **Iklim dan stok bangunan** menggeser intensitas gas secara sistematis; porsi bangunan pada periode tertentu menaikkan
  atau menurunkan EUI, yang menunjukkan kualitas selubung per era konstruksi.
* **Adopsi pemanas listrik ditentukan infrastruktur yang sudah ada.** Pangsa pemanas gas jaringan yang tinggi menekan
  adopsi; adopsi tinggi hanya muncul di kawasan yang memang kurang terhubung ke jaringan gas.
* **Sosio-demografi** menjelaskan seperempat sampai sepertiga variasi intensitas: konsumsi bukan hanya soal bangunan,
  tetapi juga siapa yang tinggal di dalamnya."""),
        code('''sub_best = pd.read_csv(OUT_CSV)
Z = explain.zone_indices(models["eui"], X, Xt, tr, te, sub_best)
q = explain.quadrant_table(Z)
explain.plot_trilemma(Z)
q.round(3)'''),
        code('''tot = dict(gas_TWh=Z.gas.sum() / 1e9, potensi_hemat_retrofit_TWh=Z.gas_saving.sum() / 1e9,
           potensi_hemat_pct=100 * Z.gas_saving.sum() / Z.gas.sum(),
           kenaikan_beban_median_pct=Z.load_increase_pct.median())
print({k: round(v, 2) for k, v in tot.items()})
cols_show = ["gas", "eui", "adopt", "gas_network_share", "saving_pct", "load_increase_pct", "kuadran"]
print("10 zona dengan potensi hemat retrofit terbesar (kWh):")
display(Z.sort_values("gas_saving", ascending=False).head(10)[cols_show].round(2))
print("10 zona dengan kenaikan beban listrik terbesar bila pemanas dielektrifikasi:")
display(Z.sort_values("load_increase_pct", ascending=False).head(10)[cols_show].round(2))'''),
        md("""### 7.2 Rekomendasi kebijakan
Trilema transisi energi (emisi, keterjangkauan, keandalan jaringan) tidak bisa diselesaikan dengan satu program
seragam. Peta di atas membagi zona menjadi empat kelompok menurut intensitas gas dan kenaikan beban listrik bila pemanas
gas beralih ke heat pump (asumsi: 80% gas rumah tangga untuk pemanasan, SCOP 3):

1. **Q1, intensitas tinggi dan beban jaringan rawan: retrofit selubung dulu, baru elektrifikasi.** Menurunkan kebutuhan
   panas lebih dulu mengurangi tagihan dan emisi sekaligus memperkecil lonjakan beban saat heat pump masuk.
2. **Q2, intensitas tinggi dan jaringan masih longgar: percepat heat pump sekarang.** Dampak emisi terbesar per rupiah
   subsidi, tanpa menunggu penguatan gardu.
3. **Q3, intensitas rendah tetapi beban jaringan rawan: perkuat gardu dan tarif fleksibel sebelum elektrifikasi.**
   Prioritas operator jaringan, bukan program bangunan.
4. **Q4: program standar berbasis pasar.**

Lintas kuadran: (a) sasaran efisiensi per m² dan sasaran emisi absolut memilih kawasan yang berbeda (hunian kecil
vs hunian besar), sehingga keduanya perlu daftar prioritas sendiri; (b) program elektrifikasi pemanas perlu menyasar
kawasan dengan pangsa gas jaringan tinggi secara aktif, karena tanpa intervensi adopsi di sana nyaris nol; (c) indeks
zona di atas dihitung dari prediksi model untuk setiap kawasan, sehingga bisa langsung dipakai sebagai daftar kerja.

Catatan kehati-hatian: SHAP menjelaskan asosiasi dalam data, bukan efek kausal; skenario beban listrik adalah
perhitungan fisik sederhana di atas prediksi model, dan makna kolom anonim adalah dugaan."""),
        md("""## 8. Kesimpulan
* Struktur data (identitas fisik, split acak per baris, adopsi per subzona) menentukan desain: satu keluarga model untuk
  gas dan EUI, identitas eksak untuk rasio bersih, dan model binomial untuk adopsi subzona baru.
* Perbaikan terbesar datang dari informasi tetangga yang dihitung jujur: encoding spasial, kNN-target di ruang stok
  bangunan, dan koreksi residual spasial; lalu dari keragaman ensemble dan blend dengan bobot bervariasi.
* Setiap keputusan model diambil dari validasi internal yang terbukti memprediksi LB publik, bukan dari LB itu sendiri.
* Interpretasi model menghasilkan peta trilema per zona yang langsung bisa dipakai sebagai urutan prioritas program
  retrofit, elektrifikasi, dan penguatan jaringan."""),
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
