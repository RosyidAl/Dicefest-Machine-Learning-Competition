"""Interpretability for the policy analysis (notebook section "Interpretabilitas dan kebijakan").

The submission ensemble mixes 40+ models and target encodings, which makes its attributions hard to read. For the
policy questions we therefore fit one transparent "explanation model" per target: a single LightGBM on observable
characteristics of the area only (base features + train/test aggregates; no target encodings, no neighbour targets),
so that every SHAP contribution refers to something a planner can measure. TreeSHAP values come from LightGBM's exact
pred_contrib. The explanation models are validated on a held-out fold, so the reader can see how faithful they are.

All functions are deterministic (fixed seeds, fixed sample, deterministic LightGBM).
"""
import numpy as np
import pandas as pd

from common import SEED, TARGET_COLS

GAS, EL, EUI, CLEAN, ADOPT = TARGET_COLS
HVAC_NAMES = {  # heating-type shares reconstructed from the household-count identity (adoption analysis)
    "hvac_dist_01": "tanpa pemanas sentral", "hvac_dist_02": "gas jaringan (sentral)", "hvac_dist_03": "tangki/LPG",
    "hvac_dist_04": "minyak", "hvac_dist_05": "kayu", "hvac_dist_06": "bahan bakar padat",
    "hvac_dist_07": "energi terbarukan", "hvac_dist_08": "jaringan panas distrik",
    "hvac_dist_09": "dua jenis+ (non-terbarukan)", "hvac_dist_10": "dua jenis+ (terbarukan)",
    "hvac_rest": "sisa pangsa (listrik + lain-lain)"}
READABLE = {"area_per_meter": "luas lantai per meter (ukuran hunian)", "bld_dim_06": "total luas lantai",
            "bld_dim_12": "jumlah meter", "bld_dim_01": "jumlah sertifikat energi", "area_per_epc": "luas per sertifikat",
            "meters_per_epc": "meter per sertifikat", "clean_ratio": "porsi bangunan bersih", "gea_ratio": "rasio luas",
            "socio_demog_016": "perkotaan (1) / perdesaan (0)", "bld_dim_18": "kategori ordinal High..Very Low (anonim)",
            **HVAC_NAMES}

TARGETS = {  # name: (label, function(train frame) -> modelled value)
    "eui": ("log intensitas gas (EUI)", lambda t: np.log(t[EUI].values)),
    "gpm": ("log gas per meter", lambda t: np.log(t[GAS].values / t["bld_dim_12"].values)),
    "epm": ("log listrik per meter", lambda t: np.log(t[EL].values / t["bld_dim_12"].values)),
    "adopt": ("adopsi pemanas listrik (%)", lambda t: t[ADOPT].values),
}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=SEED, deterministic=True,
              force_row_wise=True, verbose=-1)


def family(col):
    """feature family of a column (for the attribution shares)"""
    if col in ("area_per_meter", "area_per_epc", "gea_ratio", "a07_ratio", "a08_ratio", "a05_ratio", "a04_ratio",
               "s002_ratio", "meters_per_epc", "clean_ratio"):
        return "rasio fisik turunan"
    if col.startswith(("sub_", "zon_")) and not col.startswith(("sub_n",)):
        return "agregat kawasan"
    if col in ("zone_n", "sub_n"):
        return "lokasi"
    if col.startswith("hvac"):
        return "bauran pemanas"
    if col.startswith("bld_era"):
        return "era bangunan"
    if col.startswith("bld_dim"):
        return "dimensi bangunan"
    if col.startswith("env_"):
        return "iklim"
    if col.startswith("socio"):
        return "sosio-demografi"
    return "lainnya"


def feature_frame(d, ex, ext):
    """explanation features for train and test rows: base features + area aggregates (no target-derived columns)"""
    X = pd.concat([d.X.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)
    Xt = pd.concat([d.Xt.reset_index(drop=True), ext.reset_index(drop=True)], axis=1)
    return X, Xt


def fit_explain(X, y, rounds=800, threads=4, valid_fold=0):
    """LightGBM explanation model; returns (booster fitted on all rows, held-out R^2 of a fold-0 fit)"""
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    p = dict(PARAMS, num_threads=threads)
    a, b = list(KFold(10, shuffle=True, random_state=SEED).split(X))[valid_fold]
    m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), rounds)
    r2 = 1 - np.mean((y[b] - m.predict(X.iloc[b])) ** 2) / np.var(y[b])
    return lgb.train(p, lgb.Dataset(X, y), rounds), float(r2)


def shap_values(model, X, n=20000, seed=0):
    """exact TreeSHAP (LightGBM pred_contrib) on a fixed random sample; returns (row index, values, base value)"""
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False))
    c = model.predict(X.iloc[idx], pred_contrib=True)
    return idx, c[:, :-1], float(c[0, -1])


def family_shares(sv, cols):
    """share of mean |SHAP| per feature family"""
    imp = pd.Series(np.abs(sv).mean(0), index=cols)
    s = imp.groupby([family(c) for c in cols]).sum()
    return (s / s.sum()).sort_values(ascending=False)


def top_features(sv, cols, k=15):
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False).head(k)
    return pd.DataFrame({"fitur": imp.index, "makna (dugaan)": [READABLE.get(c, "") for c in imp.index],
                         "keluarga": [family(c) for c in imp.index], "mean |SHAP|": imp.values.round(4)})


def family_contrib(model, X, cols):
    """per-row SHAP contribution summed by family, for every row of X (used for area-level indices)"""
    c = model.predict(X, pred_contrib=True)[:, :-1]
    fam = np.array([family(col) for col in cols])
    return pd.DataFrame({f: c[:, fam == f].sum(1) for f in sorted(set(fam))})


# ---------------------------------------------------------------------------------------------------------------
# policy analysis helpers (figures and area-level indices used in the notebook)
HEAT_SHARE, SCOP = 0.8, 3.0  # scenario: 80% of domestic gas is space and water heating; heat-pump seasonal COP 3


def plot_families(summary, path=None):
    import matplotlib.pyplot as plt
    tab = pd.DataFrame({TARGETS[t][0]: pd.Series(s["families"]) for t, s in summary.items()}).fillna(0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    ax = tab.T.plot.barh(stacked=True, figsize=(8, 3.6), colormap="tab20", width=0.75)
    ax.set_xlabel("porsi mean |SHAP|")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Apa yang mendorong tiap target? (model penjelas, fitur terukur saja)")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_dependence(X, shap_res, panels, path=None):
    """panels: list of (target, feature, x label, log x?) drawn as SHAP scatter + binned mean"""
    import matplotlib.pyplot as plt
    cols = list(X.columns)
    n = len(panels)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 6.5))
    for ax, (t, feat, xlabel, logx) in zip(axs.ravel(), panels):
        idx, sv = shap_res[t][0], shap_res[t][1]
        x, s = X[feat].values[idx], sv[:, cols.index(feat)]
        ax.scatter(x, s, s=2, alpha=0.25, color="#3b6ea5", rasterized=True)
        q = pd.Series(s).groupby(pd.qcut(x, 20, duplicates="drop"), observed=True).mean()
        ax.plot([iv.mid for iv in q.index], q.values, color="#c0392b", lw=2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"SHAP ({TARGETS[t][0]})")
        if logx:
            ax.set_xscale("log")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.show()


def zone_indices(model_eui, X, Xt, tr, te, sub):
    """zone-level indicators: observed targets for train rows, final predictions for test rows.
    gas_saving: gas that would disappear if the building-stock part of log EUI above the national mean were
    removed (SHAP of the families building era + building dimensions + derived physical ratios);
    load_increase_pct: extra electricity if gas heating moved to heat pumps (HEAT_SHARE, SCOP) vs today's use."""
    allr = pd.concat([tr.assign(_src="train"), te.assign(_src="test")], ignore_index=True)
    for c in (GAS, EL, EUI, ADOPT):
        allr.loc[allr._src == "test", c] = sub[c].values
    cols = list(X.columns)
    fc = family_contrib(model_eui, pd.concat([X, Xt], ignore_index=True), cols)
    stock = fc[[c for c in ("era bangunan", "dimensi bangunan", "rasio fisik turunan") if c in fc]].sum(axis=1).values
    M = allr["bld_dim_12"].values
    g = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                      "area": allr["bld_dim_06"].values, "meters": M, "adopt_w": allr[ADOPT].values * M,
                      "gas_net_w": allr["hvac_dist_02"].values * M,
                      "gas_saving": allr[GAS].values * (1 - np.exp(-np.clip(stock, 0, None)))})
    Z = g.groupby("zone").sum()
    Z["eui"] = Z.gas / Z.area
    Z["adopt"] = Z.adopt_w / Z.meters
    Z["gas_network_share"] = Z.gas_net_w / Z.meters
    Z["load_increase_pct"] = 100 * Z.gas * HEAT_SHARE / SCOP / Z.elec
    Z["saving_pct"] = 100 * Z.gas_saving / Z.gas
    xm, ym = Z.eui.median(), Z.load_increase_pct.median()
    Z["kuadran"] = np.where(Z.eui >= xm, np.where(Z.load_increase_pct >= ym, "Q1", "Q2"),
                            np.where(Z.load_increase_pct >= ym, "Q3", "Q4"))
    return Z.drop(columns=["adopt_w", "gas_net_w"])


def quadrant_table(Z):
    q = Z.groupby("kuadran").agg(zona=("gas", "size"), gas_TWh=("gas", lambda v: v.sum() / 1e9),
                                 hemat_retrofit_TWh=("gas_saving", lambda v: v.sum() / 1e9),
                                 adopsi_rata2=("adopt", "mean"), kenaikan_beban_rata2_pct=("load_increase_pct", "mean"))
    q["porsi_gas"] = q.gas_TWh / q.gas_TWh.sum()
    q["kebijakan"] = q.index.map({"Q1": "retrofit selubung dulu, lalu elektrifikasi bertahap",
                                  "Q2": "percepat heat pump sekarang (jaringan masih longgar)",
                                  "Q3": "perkuat gardu dan tarif fleksibel sebelum elektrifikasi",
                                  "Q4": "program standar berbasis pasar"})
    return q


def plot_trilemma(Z, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(Z.eui, Z.load_increase_pct, s=8 + 300 * Z.gas / Z.gas.max(), c=Z.adopt, cmap="viridis",
                    alpha=0.75, edgecolor="k", lw=0.2)
    ax.axvline(Z.eui.median(), color="grey", ls="--", lw=0.8)
    ax.axhline(Z.load_increase_pct.median(), color="grey", ls="--", lw=0.8)
    for (x, y, t) in ((0.98, 0.98, "Q1 retrofit dulu,\nlalu elektrifikasi"), (0.98, 0.02, "Q2 heat pump\nsekarang"),
                      (0.02, 0.98, "Q3 perkuat jaringan\nsebelum elektrifikasi"), (0.02, 0.02, "Q4 program standar")):
        ax.text(x, y, t, transform=ax.transAxes, ha="right" if x > 0.5 else "left", va="top" if y > 0.5 else "bottom",
                fontsize=8, color="#444", bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.set_xlabel("intensitas gas zona (kWh/m2)")
    ax.set_ylabel("kenaikan beban listrik bila pemanas gas dielektrifikasi (%)")
    plt.colorbar(sc, label="adopsi pemanas listrik saat ini (%)")
    ax.set_title("Peta trilema per zona (ukuran titik = konsumsi gas total)")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.show()
