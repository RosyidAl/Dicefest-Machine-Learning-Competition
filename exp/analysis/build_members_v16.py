"""Write final/configs/members.yaml for v16: v15nc specs + FB07 member specs, blend lists from
exp/analysis/select_v16.json; copy the refitted correction weights to final/configs/resid_knn.json."""
import json, shutil, sys
import yaml
lam_file = sys.argv[1]
cfg = yaml.safe_load(open("final/configs/members.yaml"))
X8 = dict(learning_rate=0.03, max_depth=10, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0, eval_metric="rmse")
X4A = dict(X8, learning_rate=0.05, max_depth=8)
LP = dict(learning_rate=0.04, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
PLR = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=[1024, 512, 256], emb=dict(k=24, sigma=0.05, d=24))
PLR3 = dict(PLR, emb=dict(k=24, sigma=0.02, d=48))
TECK = dict(use_te=True, windows=[2, 10], cells=True, knn=True)
def gbm(b, kind, p, w):
    return dict(backend=b, kind=kind, params=p, wmul=w, n_folds=10, agg=True, source="src/exp_v16_knn.py", **TECK)
def nn(kind, E, w, p):
    return dict(backend="nn", kind=kind, epochs=E, fixed_ep=E - 1, wmul=w, params=p, seeds=[42, 43, 44], n_folds=10,
                agg=True, source="src/exp_v16_knn.py", **TECK)
NEW = {"xk_gpm": gbm("xgb", "gpm", X8, "iA"), "xk_lgpm": gbm("xgb", "lgas_per:bld_dim_12", X8, "A"),
       "xk_gpgea": gbm("xgb", "gas_per:bld_dim_03", X8, "iA"), "xk_lepa": gbm("xgb", "lelec_per:bld_dim_06", X4A, "M"),
       "lk_gpm": gbm("lgb", "gpm", LP, "iA"), "lk_lepa": gbm("lgb", "lelec_per:bld_dim_06", LP, "M"),
       "pk_gpm": nn("gpm", 12, "iA", PLR), "pk_gpgea": nn("gas_per:bld_dim_03", 12, "iA", PLR),
       "pk_lgpm": nn("lgas_per:bld_dim_12", 12, "A", PLR), "pk_lepm": nn("lepm", 10, "M", PLR3),
       "mk1_gpm": dict(backend="nn_multi", group="mk1", source="src/exp_v16_knn.py"),
       "mk1_epm": dict(backend="nn_multi", group="mk1", source="src/exp_v16_knn.py")}
sel = json.load(open("exp/analysis/select_v16.json"))
keep = set(sel["gas"]) | set(sel["elec"])
members = {n: s for n, s in cfg["members"].items() if n in keep}
members.update({n: s for n, s in NEW.items() if n in keep})
assert keep == set(members), keep ^ set(members)
multi = {g: m for g, m in cfg["multi"].items() if any(s.get("group") == g for s in members.values())}
if any(s.get("group") == "mk1" for s in members.values()):
    multi["mk1"] = dict(tasks=[["gpm", "gpm", "iA"], ["epm", "epm", "iM"]], epochs=12, fixed_ep=11,
                        params=dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=[2048, 1024, 512], emb=dict(k=24, sigma=0.05, d=24)),
                        seeds=[42, 43, 44], n_folds=10, agg=True, source="src/exp_v16_knn.py", **TECK)
class D(yaml.SafeDumper):
    def ignore_aliases(self, data): return True
yaml.dump(dict(members=members, multi=multi, blend=dict(gas=sel["gas"], elec=sel["elec"])),
          open("final/configs/members.yaml", "w"), Dumper=D, sort_keys=False, default_flow_style=None, width=150)
shutil.copy(lam_file, "final/configs/resid_knn.json")
print(len(members), "members;", len(sel["gas"]), "gas,", len(sel["elec"]), "elec; multi", list(multi))
