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

import notebook_text as T

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
    cells = [md(T.MODULES)]
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




def full_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [md(T.INTRO.replace("### Tim BISMILLAH", f"### Tim {TEAM}")), md(T.SETUP), code(SETUP)] + embed_cells() + [
        md(T.DATA),
        code('tr, te = load()\nprint("train", tr.shape, "| test", te.shape)\nreport.identity_checks(tr, te)'),
        md(T.TARGETS_PLOT),
        code('''fig, axs = plt.subplots(1, 4, figsize=(15, 3))
for ax, c in zip(axs, TARGET_COLS[:3] + [TARGET_COLS[4]]):
    v = tr[c].values
    ax.hist(np.log10(v) if c != TARGET_COLS[4] else v, bins=80, color="#3b6ea5")
    ax.set_title(c + (" (log10)" if c != TARGET_COLS[4] else ""), fontsize=9)
plt.tight_layout(); plt.show()
print("galat tiap target dibagi sigma berikut sebelum dirata-rata:", SIGMAS)'''),
        md(T.FEATURES),
        code('''from train_lib import Data
d = Data()
ex, ext = d.agg_features()
fam = pd.Series([explain.family(c) for c in list(d.X.columns) + list(ex.columns)]).value_counts()
print("jumlah fitur per kelompok:"); print(fam.to_string())
demo = pd.DataFrame({"log EUI": np.log(tr[TARGET_COLS[2]]), "log luas per meter": np.log(d.X["area_per_meter"]),
                     "sisa porsi pemanas": d.X["hvac_rest"], "adopsi": tr[TARGET_COLS[4]]})
print("\\nkorelasi beberapa fitur turunan dengan target:"); print(demo.corr().round(3).to_string())'''),
        md(T.ARCH),
        code('report.architecture_diagram()\nreport.member_table(cfg)'),
        md(T.TRAIN + T.TRAIN_NOTE),
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
        md(T.SUBMIT),
        code('info = pipeline.run_adoption(WORK)\nprint(info)\nsha = pipeline.assemble(WORK, OUT_CSV)\n'
             'summary = json.load(open(os.path.join(WORK, "summary.json")))\nprint("sha256", sha)'),
        md(T.EVAL),
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
        md(T.EVAL_SUMMARY),
        md(T.POLICY),
        code('''X, Xt = explain.feature_frame(d, ex, ext)
models, shap_res, summ = {}, {}, {}
for t, (label, fn) in explain.TARGETS.items():
    y = fn(tr)
    models[t], r2 = explain.fit_explain(X, y)
    shap_res[t] = explain.shap_values(models[t], X)
    summ[t] = dict(label=label, r2_holdout=round(r2, 3),
                   families=explain.family_shares(shap_res[t][1], list(X.columns)).round(4).to_dict())
pd.DataFrame({t: {"target": s["label"], "R2 pada data tak terlihat": s["r2_holdout"]} for t, s in summ.items()}).T'''),
        md(T.FAMILIES),
        code('explain.plot_families(summ)'),
        md(T.TOP),
        code('for t in ("eui", "epm", "adopt"):\n    print(explain.TARGETS[t][0]); display(explain.top_features(shap_res[t][1], list(X.columns), k=10))'),
        md(T.DEPENDENCE),
        code('''explain.plot_dependence(X, shap_res, [
    ("eui", "area_per_meter", "luas lantai per meter (m2)", True),
    ("epm", "area_per_meter", "luas lantai per meter (m2)", True),
    ("adopt", "hvac_dist_02", "porsi pemanas gas jaringan (%)", False),
    ("eui", "env_climate_05", "env_climate_05 (iklim, anonim)", False),
    ("eui", "bld_era_15", "porsi bangunan periode bld_era_15 (%)", False),
    ("eui", "bld_era_09", "porsi bangunan periode bld_era_09 (%)", False)])'''),
        md(T.DRIVERS_MAP),
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
        md(T.RECS),
        md(T.CLOSING),
        md(T.APPENDIX),
        code(T.APPX_SPLIT),
        code(T.APPX_ADV),
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
