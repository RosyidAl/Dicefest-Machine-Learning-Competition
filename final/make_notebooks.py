"""Build the two self-contained final notebooks from final/src and final/configs (the code run by run_all.py).

  final/DICEFEST2026_<TEAM>.ipynb        full pipeline: trains every model, writes the submission + sha256
  final/inference_from_artifacts.ipynb   rebuilds the submission from the saved fold models in final/artifacts

Every source file is embedded verbatim as a %%writefile cell, so each notebook alone is enough to reproduce.
usage: python final/make_notebooks.py [TEAM_NAME]
"""
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEAM_NAME", "ISI_NAMA_TIM")
SRC = ["common.py", "te.py", "train_lib.py", "nn.py", "blend.py", "adopt_v2.py", "dup_fix.py", "postprocess.py",
       "pipeline.py", "inference.py", "metric.py", "seed.py"]
CONF = ["members.yaml", "resid_knn.json"]

SETUP = '''import os, sys, platform, hashlib, json, subprocess, time
# ---- paths (edit DATA_DIR if the competition CSVs live elsewhere) ----
DATA_DIR = os.environ.get("DICEFEST_DATA_DIR", "data/raw")   # train.csv, test.csv, sample_submission.csv
ROOT = os.path.abspath("dicefest_nb")                         # embedded source code + work files
WORK = os.path.join(ROOT, "work")
OUT_CSV = os.path.join(ROOT, "submission.csv")
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
print("versions used for the reference run:", EXPECTED)'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def embed_cells():
    cells = [md("## Kode sumber\nSemua modul di bawah ini ditulis ke `dicefest_nb/src/` dan `dicefest_nb/configs/` lalu "
                "di-import. Isinya identik dengan `final/src/` dan `final/configs/` (dipakai juga oleh "
                "`final/run_all.py`).")]
    for f in SRC:
        cells.append(code(f"%%writefile dicefest_nb/src/{f}\n" + open(os.path.join(HERE, "src", f)).read()))
    for f in CONF:
        cells.append(code(f"%%writefile dicefest_nb/configs/{f}\n" + open(os.path.join(HERE, "configs", f)).read()))
    cells.append(code('sys.path.insert(0, os.path.join(ROOT, "src"))\nimport pipeline\n'
                      'cfg = pipeline.load_config()\nprint(len(cfg["members"]), "blend members")'))
    return cells


def full_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# DICE FEST 2026: The Urban Energy Transition Trilemma ({TEAM})\n\n"
           "Notebook ini mereproduksi submisi final secara deterministik dari `train.csv` dan `test.csv` saja.\n\n"
           "Alur: 50 model cross-validated (XGBoost CUDA, LightGBM, MLP PyTorch) → blend NNLS per kolom → "
           "model adopsi pemanas listrik → koreksi residual grup + kNN (bobot beku) → salin target 77 baris duplikat "
           "→ submisi + sha256.\n\n"
           "Hardware referensi: NVIDIA RTX A4000 16 GB, 16 vCPU, RAM 24 GB. Waktu jalan sekitar 20 jam "
           "(jalur GPU dan CPU paralel). Untuk verifikasi cepat dari model tersimpan, pakai "
           "`inference_from_artifacts.ipynb`."),
        code(SETUP),
    ] + embed_cells() + [
        md("## Tahap 1: melatih semua model\nTiga proses paralel: jalur `xgb` (XGBoost CUDA), `nn` (MLP PyTorch) dan "
           "`cpu` (LightGBM, 12 thread). Output: prediksi OOF dan test per model di `dicefest_nb/work/`."),
        code('''import shutil
if os.path.exists(WORK):
    shutil.rmtree(WORK)          # reproduce from a clean state
os.makedirs(WORK)
t0 = time.time()
procs = {}
for lane in pipeline.LANES:   # XGBoost (GPU), MLPs (GPU), LightGBM (CPU) in parallel
    env = dict(os.environ, OMP_NUM_THREADS=pipeline.LANE_THREADS[lane])
    cmd = [sys.executable, "-u", "-c",
           f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); import pipeline; "
           f"pipeline.run_members({lane!r}, {WORK!r})"]
    procs[lane] = subprocess.Popen(cmd, stdout=open(os.path.join(WORK, f"lane_{lane}.log"), "w"),
                                   stderr=subprocess.STDOUT, env=env)
for lane, p in procs.items():
    rc = p.wait()
    print(f"lane {lane} rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    assert rc == 0, open(os.path.join(WORK, f"lane_{lane}.log")).read()[-3000:]'''),
        md("## Tahap 2: model adopsi pemanas listrik"),
        code('info = pipeline.run_adoption(WORK)\nprint(info)'),
        md("## Tahap 3: blend, koreksi residual, salin baris duplikat, tulis submisi"),
        code('sha = pipeline.assemble(WORK, OUT_CSV)\n'
             'print(json.dumps(json.load(open(os.path.join(WORK, "summary.json")))["cv_blend_honest"], indent=1))'),
        md("## Hasil"),
        code('import pandas as pd\nsub = pd.read_csv(OUT_CSV)\nprint(sub.shape)\nprint(sub.head())\n'
             'print("sha256", hashlib.sha256(open(OUT_CSV, "rb").read()).hexdigest())'),
    ]
    return nb


def inference_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# DICE FEST 2026: submisi dari artefak model tersimpan\n\n"
           "Memuat model per fold dari `final/artifacts/models/` (checksum di `final/artifacts/SHA256SUMS`), "
           "membuat ulang prediksi OOF dan test setiap anggota tanpa training, lalu menjalankan tahap adopsi, "
           "blend, dan koreksi yang sama. Hasilnya harus identik byte dengan output notebook utama."),
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
