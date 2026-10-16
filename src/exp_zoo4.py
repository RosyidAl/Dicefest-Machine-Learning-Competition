import sys; sys.path.insert(0, "src")
from train_lib import *
d = Data()
# chained feature: census electric-heating share (train label; test = subzone lookup or model)
rest, rest_t = d.X["hvac_rest"].values, d.Xt["hvac_rest"].values
ad = d.y["target_electric_heating_adoption"]
ad_t = np.load("exp/pred/adopt_nb0.npy")
ex = pd.DataFrame({"adopt_k": ad, "r_k": rest - ad}).astype("float32")
ex_t = pd.DataFrame({"adopt_k": ad_t, "r_k": rest_t - ad_t}).astype("float32")
xp = dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=1.0, eval_metric="rmse")
run(d, "xgb_eui_wA_ad", "xgb", "eui", xp, out_dir="exp", wmul=d.A, extra=ex, extra_t=ex_t)
run(d, "xgb_epm_wM_ad", "xgb", "epm", xp, out_dir="exp", wmul=1.0 / d.M, extra=ex, extra_t=ex_t)
