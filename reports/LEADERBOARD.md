# LEADERBOARD (dibangun ulang 2026-10-16 11:26 WIB)

Semua angka NRMSE adalah OOF CV_PRIMARY (ensemble: nested/honest). Lebih kecil lebih baik.

## Terbaik global (ensemble lima target)

| exp_id | nrmse_mean | nrmse_t1 | nrmse_t2 | nrmse_t3 | hypothesis |
|---|---|---|---|---|---|
| ENS0016 | 0.25001 | 0.24590 | 0.31641 | 0.65209 | NNLS blend v15 (greedy top-25 per family, prune.py) |
| ENS0017 | 0.25002 | 0.24591 | 0.31641 | 0.65211 | NNLS blend v15nc (greedy top-25 per family, prune.py) |
| ENS0014 | 0.25005 | 0.24592 | 0.31654 | 0.65217 | NNLS blend v14 (greedy top-25 per family, prune.py) |
| ENS0015 | 0.25006 | 0.24592 | 0.31654 | 0.65219 | NNLS blend v14nc (greedy top-25 per family, prune.py) |
| ENS0012 | 0.25007 | 0.24592 | 0.31660 | 0.65219 | NNLS blend v13 (greedy top-25 per family, prune.py) |


## Terbaik per target (model tunggal)

| target | exp_id | nrmse | family | notes |
|---|---|---|---|---|
| t1 gas | E0163 | 0.24967 | mlp_plr | legacy run p2_eui_wA; log exp/v11_nn_c.log |
| t2 elec | E0166 | 0.31835 | mlp_plr | legacy run p3_epm_wM; log exp/v11_nn_e.log; member of blend v15nc |
| t3 EUI | E0162 | 0.66060 | xgboost | legacy run xs_gpm; log exp/v13_xs.log |


## Top-10 per keluarga model (t3 EUI untuk keluarga gas, t2 untuk elec)


### catboost / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0085 | 0.66163 | EUI | 68.6 | legacy run c4_eui_wA; log exp/hold_gpu.log |
| E0096 | 0.66166 | gas per meter (bld_dim_12) | 25.7 | legacy run c4_gpm; log exp/hold_gpu.log |
| E0108 | 0.66195 | gas per bld_dim_03 | 11.2 | legacy run c4_gpgea; log exp/hold_gpu.log |
| E0065 | 0.66378 | EUI | 70.9 | legacy run cat_eui_wA_tew; log exp/zoo5.log |
| E0105 | 0.66454 | log gas per meter | 36.9 | legacy run c4_lgpm; log exp/hold_gpu.log; member of blend v15 |
| E0113 | 0.66621 | log EUI | 11.3 | legacy run c4_leui_wA; log exp/hold_gpu.log; member of blend v15 |
| E0013 | 0.67182 | EUI | 14.6 | legacy run cat_eui; log exp/zoo1.log |


### catboost / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0109 | 0.32172 | elec per area | 6.9 | legacy run c4_epa; log exp/hold_gpu.log; member of blend v15 |
| E0090 | 0.32187 | elec per meter | 7.6 | legacy run c4_epm_wM; log exp/hold_gpu.log |
| E0106 | 0.32215 | log elec per area | 27.4 | legacy run c4_lepa; log exp/hold_gpu.log |
| E0068 | 0.32235 | elec per meter | 33.0 | legacy run cat_epm_wM_tew; log exp/zoo5.log |
| E0097 | 0.32236 | log elec per meter | 7.5 | legacy run c4_lepm; log exp/hold_gpu.log |
| E0014 | 0.32437 | elec per meter | 8.5 | legacy run cat_epm; log exp/zoo1.log |


### lightgbm / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0133 | 0.66165 | gas per meter (bld_dim_12) | 49.1 | legacy run la_gpm; log exp/v10_lgb.log; member of blend v15nc |
| E0072 | 0.66209 | gas per meter (bld_dim_12) | 39.5 | legacy run l4_gpm; log exp/hold_cpu.log; member of blend v15nc |
| E0135 | 0.66235 | EUI | 43.5 | legacy run la_eui_wA; log exp/v10_lgb.log |
| E0082 | 0.66262 | EUI | 31.5 | legacy run l4_eui_wA; log exp/hold_cpu.log |
| E0159 | 0.66319 | gas per bld_dim_03 | 53.6 | legacy run la_gpgea; log exp/v10_lgb_b.log |
| E0101 | 0.66332 | gas per bld_dim_03 | 40.4 | legacy run l4_gpgea; log exp/hold_cpu.log |
| E0061 | 0.66428 | gas per meter (bld_dim_12) | 48.0 | legacy run lgb_gpm_tew; log exp/lgb3.log; member of blend v15 |
| E0039 | 0.66497 | EUI | 25.6 | legacy run lgb_eui_wA_tew; log exp/lgb2.log |
| E0141 | 0.66503 | log gas per meter | 42.6 | legacy run la_lgpm; log exp/v10_lgb.log; member of blend v15nc |
| E0098 | 0.66560 | log gas per meter | 31.4 | legacy run l4_lgpm; log exp/hold_cpu.log |


### lightgbm / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0138 | 0.32143 | elec per meter | 23.4 | legacy run la_epm_wM; log exp/v10_lgb.log; member of blend v15nc |
| E0076 | 0.32161 | elec per meter | 21.3 | legacy run l4_epm_wM; log exp/hold_cpu.log; member of blend v15nc |
| E0152 | 0.32168 | elec per area | 54.5 | legacy run la_epa; log exp/v10_lgb_b.log; member of blend v15nc |
| E0103 | 0.32207 | elec per area | 25.9 | legacy run l4_epa; log exp/hold_cpu.log; member of blend v15nc |
| E0157 | 0.32217 | log elec per area | 31.8 | legacy run la_lepa; log exp/v10_lgb_b.log; member of blend v15nc |
| E0044 | 0.32229 | elec per meter | 13.3 | legacy run lgb_epm_wM_tew; log exp/lgb2.log; member of blend v15nc |
| E0146 | 0.32237 | log elec per meter | 20.8 | legacy run la_lepm; log exp/v10_lgb.log |
| E0087 | 0.32257 | log elec per meter | 37.8 | legacy run l4_lepm; log exp/hold_cpu.log |
| E0117 | 0.32263 | log elec per meter | 32.5 | legacy run l7_lepm_wv; log exp/v7_lgb.log |
| E0107 | 0.32290 | log elec per area | 22.5 | legacy run l4_lepa; log exp/hold_cpu.log; member of blend v15nc |


### mlp / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0099 | 0.66581 | gas per meter (bld_dim_12) | 19.6 | legacy run n5_gpm; log exp/v5_nn.log |
| E0067 | 0.66591 | gas per meter (bld_dim_12) | 21.7 | legacy run n4_gpm; log exp/v4_nn.log |
| E0115 | 0.66653 | gas per meter (bld_dim_12) | 9.5 | legacy run n6_gpm; log exp/hold_nn.log |
| E0119 | 0.66725 | gas per bld_dim_03 | 8.0 | legacy run n6_gpgea; log exp/hold_nn.log |
| E0078 | 0.66732 | gas per bld_dim_03 | 23.3 | legacy run n4_gpgea; log exp/v4_nn.log |
| E0123 | 0.66758 | EUI | 7.9 | legacy run n6_eui_wA; log exp/hold_nn.log |
| E0104 | 0.66759 | gas per bld_dim_03 | 19.4 | legacy run n5_gpgea; log exp/v5_nn.log |
| E0077 | 0.66766 | EUI | 23.2 | legacy run n4_eui_wA; log exp/v4_nn.log |
| E0111 | 0.66772 | gas per meter (bld_dim_12) | 11.5 | legacy run n4f_gpm; log exp/nnfix.log; member of blend v15nc |
| E0045 | 0.66861 | gas per meter (bld_dim_12) | 9.7 | legacy run nn3_gpm_te; log exp/nn4.log |


### mlp / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0116 | 0.32060 | elec per meter | 5.3 | legacy run n6_epm_wM; log exp/hold_nn.log; member of blend v15nc |
| E0122 | 0.32061 | elec per meter | 13.0 | legacy run n6b_epm_wM; log exp/v6b_nn_elec.log |
| E0073 | 0.32078 | elec per meter | 23.4 | legacy run n4_epm_wM; log exp/v4_nn.log |
| E0100 | 0.32085 | elec per meter | 19.4 | legacy run n5_epm_wM; log exp/v5_nn.log |
| E0120 | 0.32131 | log elec per meter | 6.4 | legacy run n6b_lepm; log exp/hold_nnb.log; member of blend v15nc |
| E0114 | 0.32138 | log elec per meter | 5.5 | legacy run n6_lepm; log exp/hold_nn.log |
| E0102 | 0.32144 | log elec per meter | 19.4 | legacy run n5_lepm; log exp/v5_nn.log |
| E0069 | 0.32149 | log elec per meter | 20.2 | legacy run n4_lepm; log exp/v4_nn.log |
| E0118 | 0.32173 | elec per area | 4.7 | legacy run n6_epa; log exp/hold_nn.log |
| E0124 | 0.32179 | elec per area | 11.5 | legacy run n6b_epa; log exp/v6b_nn_elec.log |


### mlp_multitask / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0167 | 0.66077 | gas per meter (bld_dim_12) |  | legacy run m1_gpm; no summary line found; member of blend v15nc |


### mlp_multitask / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0168 | 0.31910 | elec per meter |  | legacy run m1_epm; no summary line found; member of blend v15nc |


### mlp_plr / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0140 | 0.66084 | gas per meter (bld_dim_12) | 34.2 | legacy run p2_gpm; log exp/v11_nn.log; member of blend v15nc |
| E0148 | 0.66097 | gas per bld_dim_03 | 50.0 | legacy run p2_gpgea; log exp/v11_nn_b.log; member of blend v15nc |
| E0163 | 0.66097 | EUI | 81.2 | legacy run p2_eui_wA; log exp/v11_nn_c.log |
| E0145 | 0.66421 | log gas per meter | 56.1 | legacy run p2_lgpm; log exp/v11_nn_b.log; member of blend v15nc |
| E0150 | 0.66746 | log EUI | 50.1 | legacy run p2_leui_wA; log exp/v11_nn_b.log |
| E0155 | 0.66777 | gas kWh | 57.2 | legacy run p2_gas; log exp/v11_nn_c.log |
| E0151 | 0.67561 | gas per EPC count (bld_dim_01) | 51.1 | legacy run p2_gpn; log exp/v11_nn_c.log |
| E0134 | 0.67981 | gas per meter (bld_dim_12) | 190.6 | legacy run p1_gpm; log exp/v9_nn_c.log; member of blend v15nc |


### mlp_plr / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0166 | 0.31835 | elec per meter | 75.7 | legacy run p3_epm_wM; log exp/v11_nn_e.log; member of blend v15nc |
| E0147 | 0.31902 | elec per meter | 50.6 | legacy run p2_epm_wM; log exp/v11_nn_b.log |
| E0161 | 0.31931 | log elec per meter | 97.7 | legacy run p3_lepm; log exp/v11_nn_e.log; member of blend v15nc |
| E0160 | 0.31981 | elec per area | 75.9 | legacy run p2_epa; log exp/v11_nn_c.log |
| E0137 | 0.32025 | log elec per meter | 35.1 | legacy run p2_lepm; log exp/v11_nn.log |
| E0158 | 0.32063 | elec kWh | 73.7 | legacy run p2_elec; log exp/v11_nn_c.log |
| E0131 | 0.32530 | elec per meter | 96.7 | legacy run p1_epm_wM; log exp/v9_nn_b.log |
| E0130 | 0.32713 | log elec per meter | 94.7 | legacy run p1_lepm; log exp/v9_nn.log |


### tabm_mini_plr / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0153 | 0.66136 | gas per meter (bld_dim_12) | 116.1 | legacy run t1_gpm; log exp/v11_nn.log |


### tabm_mini_plr / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0164 | 0.31938 | elec per meter | 265.2 | legacy run t1_epm_wM; log exp/v11_nn_d.log |
| E0149 | 0.31992 | log elec per meter | 176.4 | legacy run t1_lepm; log exp/v11_nn.log; member of blend v15nc |


### xgboost / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0162 | 0.66060 | gas per meter (bld_dim_12) | 84.6 | legacy run xs_gpm; log exp/v13_xs.log |
| E0132 | 0.66094 | gas per meter (bld_dim_12) | 75.9 | legacy run xa_gpm; log exp/v10_agg.log; member of blend v15nc |
| E0126 | 0.66131 | gas per meter (bld_dim_12) | 18.8 | legacy run x8_gpm; log exp/v8_xgb.log; member of blend v15nc |
| E0136 | 0.66294 | gas per bld_dim_03 | 38.2 | legacy run xa_gpgea; log exp/v10_agg_b.log; member of blend v15nc |
| E0091 | 0.66332 | gas per meter (bld_dim_12) | 23.9 | legacy run x4_gpm; log exp/hold_gpu.log |
| E0127 | 0.66351 | gas per bld_dim_03 | 15.7 | legacy run x8_gpgea; log exp/v8_xgb.log |
| E0139 | 0.66465 | log gas per meter | 36.1 | legacy run xa_lgpm; log exp/v10_agg_b.log; member of blend v15nc |
| E0095 | 0.66513 | gas per bld_dim_03 | 7.0 | legacy run x4_gpgea; log exp/hold_xgbrev.log |
| E0128 | 0.66525 | log gas per meter | 15.9 | legacy run x8_lgpm; log exp/v8_xgb.log; member of blend v15nc |
| E0074 | 0.66536 | EUI | 16.1 | legacy run x4_eui_wA; log exp/hold_xgbrev.log |


### xgboost / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0081 | 0.32197 | elec per meter | 4.6 | legacy run x4_epm_wM; log exp/hold_xgbrev.log; member of blend v15nc |
| E0143 | 0.32199 | elec per meter | 25.9 | legacy run xa_epm_wM; log exp/v10_agg_b.log |
| E0066 | 0.32208 | elec per meter | 15.9 | legacy run xgb_epm_wM_tew_k10; log exp/k10.log; member of blend v15nc |
| E0165 | 0.32212 | elec per meter | 31.5 | legacy run xs_epm_wM; log exp/v13_xs.log |
| E0142 | 0.32268 | log elec per area | 24.6 | legacy run xa_lepa; log exp/v10_agg_b.log; member of blend v15nc |
| E0058 | 0.32276 | elec per meter | 7.2 | legacy run xgb_epm_wM_tewc; log exp/cell1.log |
| E0089 | 0.32291 | elec per area | 4.5 | legacy run x4_epa; log exp/hold_xgbrev.log; member of blend v15nc |
| E0023 | 0.32291 | elec per meter | 1.9 | legacy run xgb_epm_wM_te; log exp/zoo3.log |
| E0075 | 0.32293 | log elec per area | 5.8 | legacy run x4_lepa; log exp/hold_gpu.log; member of blend v15nc |
| E0024 | 0.32294 | elec per meter | 2.1 | legacy run xgb_epm_wM_tew; log exp/zoo3.log |


### xgboost_linear / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0070 | 0.66815 | EUI | 64.0 | legacy run xgbl_eui_wA_tew; log exp/zoo6.log |


## 20 eksperimen terakhir

| exp_id | status | model_family | nrmse_mean | nrmse_t1 | nrmse_t2 | nrmse_t3 | hypothesis |
|---|---|---|---|---|---|---|---|
| ENS0002 | done | ensemble | 0.25061 | 0.24659 | 0.31753 | 0.65329 | NNLS blend v8 (greedy top-25 per family, prune.py) |
| ENS0003 | done | ensemble | 0.25066 | 0.24666 | 0.31758 | 0.65340 | NNLS blend v8_nocat (greedy top-25 per family, prune.py) |
| ENS0004 | done | ensemble | 0.25055 | 0.24659 | 0.31722 | 0.65329 | NNLS blend v9 (greedy top-25 per family, prune.py) |
| ENS0005 | done | ensemble | 0.25059 | 0.24666 | 0.31725 | 0.65340 | NNLS blend v9nc (greedy top-25 per family, prune.py) |
| ENS0006 | done | ensemble | 0.25055 | 0.24659 | 0.31720 | 0.65329 | NNLS blend v10 (greedy top-25 per family, prune.py) |
| ENS0007 | done | ensemble | 0.25059 | 0.24666 | 0.31723 | 0.65340 | NNLS blend v10nc (greedy top-25 per family, prune.py) |
| ENS0008 | done | ensemble | 0.25034 | 0.24623 | 0.31720 | 0.65262 | NNLS blend v11 (greedy top-25 per family, prune.py) |
| ENS0009 | done | ensemble | 0.25037 | 0.24629 | 0.31723 | 0.65268 | NNLS blend v11nc (greedy top-25 per family, prune.py) |
| ENS0010 | done | ensemble | 0.25015 | 0.24602 | 0.31683 | 0.65227 | NNLS blend v12 (greedy top-25 per family, prune.py) |
| ENS0011 | done | ensemble | 0.25017 | 0.24604 | 0.31684 | 0.65230 | NNLS blend v12nc (greedy top-25 per family, prune.py) |
| ENS0012 | done | ensemble | 0.25007 | 0.24592 | 0.31660 | 0.65219 | NNLS blend v13 (greedy top-25 per family, prune.py) |
| ENS0013 | done | ensemble | 0.25008 | 0.24593 | 0.31660 | 0.65221 | NNLS blend v13nc (greedy top-25 per family, prune.py) |
| ENS0014 | done | ensemble | 0.25005 | 0.24592 | 0.31654 | 0.65217 | NNLS blend v14 (greedy top-25 per family, prune.py) |
| ENS0015 | done | ensemble | 0.25006 | 0.24592 | 0.31654 | 0.65219 | NNLS blend v14nc (greedy top-25 per family, prune.py) |
| ENS0016 | done | ensemble | 0.25001 | 0.24590 | 0.31641 | 0.65209 | NNLS blend v15 (greedy top-25 per family, prune.py) |
| ENS0017 | done | ensemble | 0.25002 | 0.24591 | 0.31641 | 0.65211 | NNLS blend v15nc (greedy top-25 per family, prune.py) |
| PP0001 | done | postprocess |  |  |  |  | group residual correction (zone, climate cell, +-30 sub_n window), lambda fitted on holdout |
| PP0002 | done | postprocess |  |  |  |  | adoption v2: binomial q model with exact household count N |
| PP0003 | done | postprocess |  |  |  |  | train-duplicate copy (77 test rows identical to a train row) |
| PP0004 | done | postprocess |  |  |  |  | similarity-kNN residual correction on top of PP0001 |


## Submisi dengan skor public LB

| created_wib | file | cv_mean | public_lb | note |
|---|---|---|---|---|
| 2026-10-13 19:52 | submissions/sub_v6pre.csv |  | 0.25125 | submitted by user |
| 2026-10-13 20:33 | submissions/sub_probe_x4.csv |  | 0.25317 | submitted by user |
| 2026-10-14 04:29 | submissions/sub_v8.csv | 0.25061 | 0.25116 | submitted by user |
| 2026-10-14 12:58 | submissions/sub_v8nc.csv | 0.25066 | 0.25118 | submitted by user |
| 2026-10-14 17:19 | submissions/sub_v9r.csv |  | 0.25070 | submitted by user |
| 2026-10-14 22:08 | submissions/sub_v12r.csv | 0.25015 (pre-corr) | 0.25025 | submitted by user |
| 2026-10-15 03:03 | submissions/sub_v13ra.csv | 0.25007 (pre-corr) | 0.25008 | submitted by user |
| 2026-10-15 08:21 | submissions/sub_v15rd.csv | 0.25001 (pre-corr) | 0.25000 | submitted by user |
| 2026-10-15 08:22 | submissions/sub_v15ncrd.csv | 0.25002 (pre-corr) | 0.2499 | submitted by user |
