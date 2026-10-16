"""Rebuild reports/LEADERBOARD.md and reports/dashboard.html from experiments/registry.csv and the submission log."""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tracking import now_wib, read_registry  # noqa: E402

T = {"nrmse_t1": "t1 gas", "nrmse_t2": "t2 elec", "nrmse_t3": "t3 EUI", "nrmse_t4": "t4 clean", "nrmse_t5": "t5 adopt"}


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def md_table(df, cols):
    if df.empty:
        return "_(kosong)_\n"
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


def main():
    reg = read_registry()
    done = reg[reg["status"] == "done"].copy()
    for c in ["nrmse_mean", *T]:
        done[c + "_f"] = num(done, c)
    lines = [f"# LEADERBOARD (dibangun ulang {now_wib()} WIB)\n",
             "Semua angka NRMSE adalah OOF CV_PRIMARY (ensemble: nested/honest). Lebih kecil lebih baik.\n",
             "## Terbaik global (ensemble lima target)\n"]
    ens = done[done["nrmse_mean_f"].notna()].sort_values("nrmse_mean_f").head(5)
    lines.append(md_table(ens, ["exp_id", "nrmse_mean", "nrmse_t1", "nrmse_t2", "nrmse_t3", "hypothesis"]))
    lines.append("\n## Terbaik per target (model tunggal)\n")
    single = done[~done["model_family"].isin(["ensemble", "postprocess"])]
    rows = []
    for c, lab in T.items():
        s = single[single[c + "_f"].notna()].sort_values(c + "_f").head(1)
        for _, r in s.iterrows():
            rows.append({"target": lab, "exp_id": r["exp_id"], "nrmse": r[c], "family": r["model_family"],
                         "notes": r["notes"][:80]})
    lines.append(md_table(pd.DataFrame(rows), ["target", "exp_id", "nrmse", "family", "notes"]))
    lines.append("\n## Top-10 per keluarga model (t3 EUI untuk keluarga gas, t2 untuk elec)\n")
    for fam, g in single.groupby("model_family"):
        for c in ("nrmse_t3", "nrmse_t2"):
            gg = g[g[c + "_f"].notna()].sort_values(c + "_f").head(10)
            if len(gg):
                lines.append(f"\n### {fam} / {T[c]}\n")
                lines.append(md_table(gg, ["exp_id", c, "target_transform", "runtime_min", "notes"]))
    lines.append("\n## 20 eksperimen terakhir\n")
    last = reg.tail(20)
    lines.append(md_table(last, ["exp_id", "status", "model_family", "nrmse_mean", "nrmse_t1", "nrmse_t2", "nrmse_t3",
                                 "hypothesis"]))
    if os.path.exists("submissions/submission_log.csv"):
        sl = pd.read_csv("submissions/submission_log.csv", dtype=str, keep_default_na=False)
        lines.append("\n## Submisi dengan skor public LB\n")
        lines.append(md_table(sl[sl["public_lb"] != ""], ["created_wib", "file", "cv_mean", "public_lb", "note"]))
    open("reports/LEADERBOARD.md", "w").write("\n".join(lines))

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Ensemble honest CV vs waktu", "Model tunggal t3 EUI per keluarga",
                                                        "Model tunggal t2 elec per keluarga", "CV vs public LB"])
    e = done[done["model_family"] == "ensemble"]
    fig.add_trace(go.Scatter(x=e["exp_id"], y=e["nrmse_mean_f"], mode="lines+markers", name="ensemble"), 1, 1)
    for c, (r_, c_) in (("nrmse_t3", (1, 2)), ("nrmse_t2", (2, 1))):
        for fam, g in single[single[c + "_f"].notna()].groupby("model_family"):
            fig.add_trace(go.Box(y=g[c + "_f"], name=fam, showlegend=False), r_, c_)
    if os.path.exists("submissions/submission_log.csv"):
        s = sl[sl["public_lb"] != ""].copy()
        s["cv"] = pd.to_numeric(s["cv_mean"].str.split().str[0], errors="coerce")
        s["lb"] = pd.to_numeric(s["public_lb"], errors="coerce")
        fig.add_trace(go.Scatter(x=s["cv"], y=s["lb"], text=s["file"], mode="markers", name="CV vs LB"), 2, 2)
    fig.update_layout(height=900, title=f"DICE FEST 2026 dashboard ({now_wib()} WIB)")
    fig.write_html("reports/dashboard.html", include_plotlyjs=True)
    print("reports rebuilt")


if __name__ == "__main__":
    main()
