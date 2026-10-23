"""Small helpers that make the notebook readable: data identity checks, a member table and an architecture diagram."""
import numpy as np
import pandas as pd

from common import TARGET_COLS

GAS, EL, EUI, CLEAN, ADOPT = TARGET_COLS


def identity_checks(tr, te):
    """the physical / structural identities the pipeline relies on, checked on the data"""
    eui_gas = np.abs(tr[EUI] * tr["bld_dim_06"] / tr[GAS] - 1).max()
    clean = (np.round(100 * tr["bld_dim_02"] / tr["bld_dim_01"], 2) == tr[CLEAN]).mean()
    adopt_const = (tr.groupby("subzone_id")[ADOPT].nunique() == 1).mean()
    n = len(tr) + len(te)
    return pd.DataFrame([
        ("EUI x luas lantai (bld_dim_06) = gas", f"galat relatif maks {eui_gas:.1e}"),
        ("rasio bangunan bersih = round(100 x bld_dim_02 / bld_dim_01, 2)", f"tepat di {clean:.2%} baris train"),
        ("adopsi pemanas listrik konstan per subzona", f"{adopt_const:.2%} subzona"),
        ("porsi test dari semua baris", f"{len(te) / n:.6f} (split acak 70/30 per baris)"),
        ("baris test di subzona yang ada di train", f"{te.subzone_id.isin(set(tr.subzone_id)).mean():.2%}"),
    ], columns=["identitas / struktur", "hasil cek"])


def split_checks(tr, te):
    """how the test rows were drawn and what a subzone is, from features only"""
    n = pd.concat([tr.subzone_id, te.subzone_id]).value_counts().values
    expected = (n * 0.3 * 0.3 ** (n - 1)).sum() / (0.3 * n.sum())   # P(no train row in the subzone) under a 30% row split
    observed = 1 - te.subzone_id.isin(set(tr.subzone_id)).mean()
    rows = pd.concat([tr[["zone_region_id", "subzone_id"]], te[["zone_region_id", "subzone_id"]]])
    order = np.argsort(rows.subzone_id.str[4:].astype(int).values)
    zone = rows.zone_region_id.str[4:].astype(int).values[order]
    census = [c for c in sorted(tr.columns) if c.startswith(("socio_demog_", "hvac_dist_", "env_climate_"))
              and pd.api.types.is_numeric_dtype(tr[c]) and tr[c].astype(float).var() > 0]
    def within(c):   # share of the variance left after removing the subzone mean
        v = tr[c].astype(float)
        return (v - v.groupby(tr.subzone_id).transform("mean")).var() / v.var()

    return pd.DataFrame([
        ("porsi kawasan uji di subzona baru", f"{observed:.2%} (harapan pengacakan per baris {expected:.2%})"),
        ("baris berurutan menurut nomor subzona yang berada di zona sama", f"{np.mean(zone[1:] == zone[:-1]):.2%}"),
        ("kolom sensus yang konstan di dalam subzona", f"{sum(within(c) < 1e-12 for c in census)} dari {len(census)}"),
    ], columns=["pemeriksaan", "hasil"])


def adversarial_auc(tr, te, threads=8):
    """5-fold LightGBM that tries to tell train rows from test rows; AUC near 0.5 means a random split"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    feat = sorted(c for c in tr.columns if c.startswith(("bld_dim_", "bld_era_", "hvac_dist_", "env_climate_", "socio_demog_")))
    X = pd.concat([tr[feat], te[feat]], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, subsample=0.5, subsample_freq=1,
                               colsample_bytree=0.5, num_threads=threads, seed=42, deterministic=True,
                               force_row_wise=True, verbose=-1).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    return roc_auc_score(y, oof)


def member_table(cfg):
    """one row per blend member: backend, target formulation, weights, feature blocks"""
    rows = []
    blend = set(cfg["blend"]["gas"]) | set(cfg["blend"]["elec"])
    for name, s in cfg["members"].items():
        if name not in blend:
            continue
        fam = "listrik" if name in cfg["blend"]["elec"] else "gas / EUI"
        if s["backend"] == "nn_multi":
            m = cfg["multi"][s["group"]]
            rows.append((name, "MLP gas + listrik", fam, "gas dan listrik per meter", "-",
                         "dasar + rata-rata wilayah" + (" + kawasan mirip" if m.get("knn") else "")))
            continue
        parts = ["dasar"]
        if s.get("use_te"):
            parts.append("rata-rata wilayah")
        if s.get("knn"):
            parts.append("kawasan mirip" + (" (beberapa sudut pandang)" if s.get("knn_x") else ""))
        if s.get("agg"):
            parts.append("gambaran kawasan")
        feats = " + ".join(parts)
        backend = {"xgb": "XGBoost", "lgb": "LightGBM", "nn": "MLP (PLR/TabM)" if s.get("params", {}).get("emb") else "MLP",
                   "nca": "NCA (cari tetangga)", "cat": "CatBoost"}[s["backend"]]
        obj = s.get("params", {}).get("objective", "")
        if "poisson" in str(obj):
            backend += " Poisson"
        rows.append((name, backend, fam, readable_kind(s["kind"]), WEIGHTS.get(s.get("wmul"), "-"), feats))
    return pd.DataFrame(rows, columns=["anggota", "model", "keluarga", "yang ditebak", "bobot baris", "fitur"])


WEIGHTS = {"iA": "1 / luas", "A": "luas", "iM": "1 / meter", "M": "meter", "sqrtM": "akar meter", None: "-"}


def readable_kind(kind):
    """target formulation of a member in words"""
    base = {"gpm": "gas per meter", "epm": "listrik per meter", "lepm": "log listrik per meter", "eui": "EUI",
            "leui": "log EUI", "gas": "gas total", "elec": "listrik total", "epa": "listrik per luas"}
    if kind in base:
        return base[kind]
    if ":" in kind:
        fam, col = kind.split(":")
        unit = {"bld_dim_12": "meter", "bld_dim_06": "luas", "bld_dim_03": "luas selubung", "bld_dim_01": "sertifikat"}.get(col, col)
        what = "gas" if "gas" in fam else "listrik"
        return ("log " if fam.startswith("l") else "") + f"{what} per {unit}"
    return kind


def architecture_diagram(path=None):
    """block diagram of the multi-output pipeline"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333", lw=0.9))

    box(0.1, 2.2, 1.6, 0.8, "train.csv + test.csv\n(201 kolom)", "#eeeeee")
    box(2.1, 3.5, 2.2, 1.2, "fitur domain\nukuran rumah, gambaran kawasan,\nrata-rata wilayah (zona, petak iklim,\ntetangga), kawasan paling mirip", "#dbe9f6")
    box(2.1, 0.4, 2.2, 1.2, "rumus eksak\nporsi bangunan bersih = b02/b01\nadopsi: salin per subzona\n+ model peluang (N rumah)", "#e2f0d9")
    box(4.7, 3.5, 2.3, 1.2, "41 model: 28 untuk gas/EUI,\n13 untuk listrik\nXGBoost, LightGBM, jaringan\nsaraf, NCA; tebakan per rumah", "#fde9d9")
    box(7.4, 3.5, 1.7, 1.2, "gabungan model\nbobot menyesuaikan\n(luas, meter,\nbesar tebakan)", "#fff2cc")
    box(7.4, 1.8, 1.7, 1.2, "koreksi galat\nwilayah dan\ntetangga mirip", "#fff2cc")
    box(9.4, 1.8, 1.5, 1.2, "penyesuaian akhir\nmeter sedikit, batas\ngas 1 juta kWh,\nsalin duplikat", "#fff2cc")
    box(9.4, 0.3, 1.5, 0.9, "file submisi\n5 kolom + sha256", "#eeeeee")
    arrow(1.7, 2.6, 2.1, 4.0)
    arrow(1.7, 2.6, 2.1, 1.0)
    arrow(4.3, 4.1, 4.7, 4.1)
    arrow(7.0, 4.1, 7.4, 4.1)
    arrow(8.25, 3.5, 8.25, 3.0)
    arrow(9.1, 2.4, 9.4, 2.4)
    arrow(10.15, 1.8, 10.15, 1.2)
    arrow(4.3, 1.0, 9.4, 0.75)
    ax.text(5.5, 2.75, "gas = EUI x luas lantai: satu keluarga model\nmelayani dua kolom",
            fontsize=8, ha="center", style="italic", color="#555")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.show()
