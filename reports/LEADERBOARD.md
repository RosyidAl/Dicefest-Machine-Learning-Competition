# LEADERBOARD (dibangun ulang 2026-10-19 22:53 WIB)

Semua angka NRMSE adalah OOF CV_PRIMARY (ensemble: nested/honest). Lebih kecil lebih baik.

## Terbaik global (ensemble lima target)

| exp_id | nrmse_mean | nrmse_t1 | nrmse_t2 | nrmse_t3 | hypothesis |
|---|---|---|---|---|---|
| ENS0018 | 0.24960 | 0.24524 | 0.31608 | 0.65101 | NNLS blend v16 (34 anggota, FB07 kNN-target; exp/analysis/select_v16.py) = Final A (D006) |
| ENS0016 | 0.25001 | 0.24590 | 0.31641 | 0.65209 | NNLS blend v15 (greedy top-25 per family, prune.py) |
| ENS0017 | 0.25002 | 0.24591 | 0.31641 | 0.65211 | NNLS blend v15nc (greedy top-25 per family, prune.py) |
| ENS0014 | 0.25005 | 0.24592 | 0.31654 | 0.65217 | NNLS blend v14 (greedy top-25 per family, prune.py) |
| ENS0015 | 0.25006 | 0.24592 | 0.31654 | 0.65219 | NNLS blend v14nc (greedy top-25 per family, prune.py) |


## Terbaik per target (model tunggal)

| target | exp_id | nrmse | family | notes |
|---|---|---|---|---|
| t1 gas | EM0003 | 0.24873 | xgb | member xp_eui; v17 selection: blend total -0.000035 with all four, not adopted ( |
| t2 elec | E0166 | 0.31835 | mlp_plr | legacy run p3_epm_wM; log exp/v11_nn_e.log; member of blend v15nc |
| t3 EUI | EM0002 | 0.65741 | xgb | member xp_gpm; v17 selection: blend total -0.000035 with all four, not adopted ( |
| t4 clean | E0173 | 0.01575 | xgb |  |
| t5 adopt | E0209 | 0.07481 | lgb |  screen_folds=3 |


## Top-10 per keluarga model (t3 EUI untuk keluarga gas, t2 untuk elec)


### catboost / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0085 | 0.66163 | EUI | 127.6 | legacy run c4_eui_wA; log exp/v4_cat.log |
| E0096 | 0.66166 | gas per meter (bld_dim_12) | 35.8 | legacy run c4_gpm; log exp/v4_cat.log |
| E0108 | 0.66195 | gas per bld_dim_03 | 16.4 | legacy run c4_gpgea; log exp/v5_cat.log |
| E0065 | 0.66378 | EUI | 70.9 | legacy run cat_eui_wA_tew; log exp/zoo5.log |
| E0105 | 0.66454 | log gas per meter | 79.5 | legacy run c4_lgpm; log exp/v5_cat.log; member of blend v15 |
| E0113 | 0.66621 | log EUI | 25.4 | legacy run c4_leui_wA; log exp/v5_cat.log; member of blend v15 |
| E0013 | 0.67182 | EUI | 14.6 | legacy run cat_eui; log exp/zoo1.log |


### catboost / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0109 | 0.32172 | elec per area | 16.6 | legacy run c4_epa; log exp/v5_cat.log; member of blend v15 |
| E0090 | 0.32187 | elec per meter | 25.6 | legacy run c4_epm_wM; log exp/v4_cat.log |
| E0106 | 0.32215 | log elec per area | 10.3 | legacy run c4_lepa; log exp/v5_cat.log |
| E0068 | 0.32235 | elec per meter | 33.0 | legacy run cat_epm_wM_tew; log exp/zoo5.log |
| E0097 | 0.32236 | log elec per meter | 10.7 | legacy run c4_lepm; log exp/v4_cat.log |
| E0014 | 0.32437 | elec per meter | 8.5 | legacy run cat_epm; log exp/zoo1.log |


### lgb / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0262 | 0.66536 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}} | 29.9 |  screen_folds=3 |
| E0261 | 0.66755 | {"default": {"type": "identity"}} | 31.7 |  screen_folds=3 |
| E0207 | 0.66947 | {"default": {"type": "identity"}} | 17.7 |  screen_folds=3 |
| E0273 | 0.6699013376971004 | gpm (ratio to bld_dim_12) | 16.4 | screening lr 0.1, early stopping on the validation fold (relative comparison only) |
| E0263 | 0.67030 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}} | 15.9 |  screen_folds=3 |
| E0260 | 0.67119 | {"target_macro_elec_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_12", "weight_col_power": -1, "weight_power": 2}, "target_macro_gas_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}} | 17.2 |  screen_folds=3 |
| E0271 | 0.6720985399324281 | gpm (ratio to bld_dim_12) | 5.7 | screening lr 0.1, early stopping on the validation fold (relative comparison only) |
| E0208 | 0.67357 | {"default": {"type": "log1p"}} | 12.0 |  screen_folds=3 |
| E0205 | 0.67381 | {"default": {"type": "identity"}} | 9.1 |  screen_folds=3 |
| E0202 | 0.67432 | {"default": {"type": "sqrt"}} | 9.5 |  screen_folds=3 |


### lgb / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0262 | 0.32385 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}} | 29.9 |  screen_folds=3 |
| E0207 | 0.32543 | {"default": {"type": "identity"}} | 17.7 |  screen_folds=3 |
| E0261 | 0.32543 | {"default": {"type": "identity"}} | 31.7 |  screen_folds=3 |
| E0274 | 0.32545917304677086 | epm (ratio to bld_dim_12) | 54.3 | screening lr 0.1, early stopping on the validation fold (relative comparison only) |
| E0263 | 0.32587 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}} | 15.9 |  screen_folds=3 |
| E0272 | 0.3262256196151608 | epm (ratio to bld_dim_12) | 5.1 | screening lr 0.1, early stopping on the validation fold (relative comparison only) |
| E0260 | 0.32633 | {"target_macro_elec_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_12", "weight_col_power": -1, "weight_power": 2}, "target_macro_gas_kwh": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}} | 17.2 |  screen_folds=3 |
| E0204 | 0.32675 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 2}} | 6.4 |  screen_folds=3 |
| E0202 | 0.32776 | {"default": {"type": "sqrt"}} | 9.5 |  screen_folds=3 |
| E0205 | 0.32791 | {"default": {"type": "identity"}} | 9.1 |  screen_folds=3 |


### lightgbm / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0133 | 0.66165 | gas per meter (bld_dim_12) | 49.1 | legacy run la_gpm; log exp/v10_lgb.log; member of blend v15nc |
| E0072 | 0.66209 | gas per meter (bld_dim_12) | 77.4 | legacy run l4_gpm; log exp/v4_lgb.log; member of blend v15nc |
| E0135 | 0.66235 | EUI | 43.5 | legacy run la_eui_wA; log exp/v10_lgb.log |
| E0082 | 0.66262 | EUI | 65.2 | legacy run l4_eui_wA; log exp/v4_lgb.log |
| E0159 | 0.66319 | gas per bld_dim_03 | 53.6 | legacy run la_gpgea; log exp/v10_lgb_b.log |
| E0101 | 0.66332 | gas per bld_dim_03 | 43.4 | legacy run l4_gpgea; log exp/v5_lgb.log |
| E0061 | 0.66428 | gas per meter (bld_dim_12) | 48.0 | legacy run lgb_gpm_tew; log exp/lgb3.log; member of blend v15 |
| E0039 | 0.66497 | EUI | 25.6 | legacy run lgb_eui_wA_tew; log exp/lgb2.log |
| E0141 | 0.66503 | log gas per meter | 42.6 | legacy run la_lgpm; log exp/v10_lgb.log; member of blend v15nc |
| E0098 | 0.66560 | log gas per meter | 32.8 | legacy run l4_lgpm; log exp/v4_lgb.log |


### lightgbm / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0138 | 0.32143 | elec per meter | 23.4 | legacy run la_epm_wM; log exp/v10_lgb.log; member of blend v15nc |
| E0076 | 0.32161 | elec per meter | 34.4 | legacy run l4_epm_wM; log exp/v4_lgb.log; member of blend v15nc |
| E0152 | 0.32168 | elec per area | 54.5 | legacy run la_epa; log exp/v10_lgb_b.log; member of blend v15nc |
| E0103 | 0.32207 | elec per area | 27.4 | legacy run l4_epa; log exp/v5_lgb.log; member of blend v15nc |
| E0157 | 0.32217 | log elec per area | 31.8 | legacy run la_lepa; log exp/v10_lgb_b.log; member of blend v15nc |
| E0044 | 0.32229 | elec per meter | 13.3 | legacy run lgb_epm_wM_tew; log exp/lgb2.log; member of blend v15nc |
| E0146 | 0.32237 | log elec per meter | 20.8 | legacy run la_lepm; log exp/v10_lgb.log |
| E0087 | 0.32257 | log elec per meter | 27.6 | legacy run l4_lepm; log exp/v4_lgb.log |
| E0117 | 0.32263 | log elec per meter | 32.5 | legacy run l7_lepm_wv; log exp/v7_lgb.log |
| E0107 | 0.32290 | log elec per area | 24.9 | legacy run l4_lepa; log exp/v5_lgb.log; member of blend v15nc |


### mean / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0169 | 1.00006 | {"default": {"type": "identity"}} | 0.1 |  |
| E0179 | 1.00088 | {"default": {"type": "identity"}} | 0.1 |  |


### mean / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0169 | 1.00000 | {"default": {"type": "identity"}} | 0.1 |  |
| E0179 | 1.00022 | {"default": {"type": "identity"}} | 0.1 |  |


### mlp / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0099 | 0.66581 | gas per meter (bld_dim_12) | 19.6 | legacy run n5_gpm; log exp/v5_nn.log |
| E0067 | 0.66591 | gas per meter (bld_dim_12) | 21.7 | legacy run n4_gpm; log exp/v4_nn.log |
| E0115 | 0.66653 | gas per meter (bld_dim_12) | 22.2 | legacy run n6_gpm; log exp/v6_nn_gas.log |
| E0119 | 0.66725 | gas per bld_dim_03 | 18.8 | legacy run n6_gpgea; log exp/v6_nn_gas.log |
| E0078 | 0.66732 | gas per bld_dim_03 | 23.3 | legacy run n4_gpgea; log exp/v4_nn.log |
| E0123 | 0.66758 | EUI | 19.1 | legacy run n6_eui_wA; log exp/v6_nn_gas.log |
| E0104 | 0.66759 | gas per bld_dim_03 | 19.4 | legacy run n5_gpgea; log exp/v5_nn.log |
| E0077 | 0.66766 | EUI | 23.2 | legacy run n4_eui_wA; log exp/v4_nn.log |
| E0111 | 0.66772 | gas per meter (bld_dim_12) | 11.5 | legacy run n4f_gpm; log exp/nnfix.log; member of blend v15nc |
| E0045 | 0.66861 | gas per meter (bld_dim_12) | 9.7 | legacy run nn3_gpm_te; log exp/nn4.log |


### mlp / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0116 | 0.32060 | elec per meter | 13.4 | legacy run n6_epm_wM; log exp/v6_nn_elec.log; member of blend v15nc |
| E0122 | 0.32061 | elec per meter | 13.0 | legacy run n6b_epm_wM; log exp/v6b_nn_elec.log |
| E0073 | 0.32078 | elec per meter | 23.4 | legacy run n4_epm_wM; log exp/v4_nn.log |
| E0100 | 0.32085 | elec per meter | 19.4 | legacy run n5_epm_wM; log exp/v5_nn.log |
| E0120 | 0.32131 | log elec per meter | 12.9 | legacy run n6b_lepm; log exp/v6b_nn_elec.log; member of blend v15nc |
| E0114 | 0.32138 | log elec per meter | 12.8 | legacy run n6_lepm; log exp/v6_nn_elec.log |
| E0102 | 0.32144 | log elec per meter | 19.4 | legacy run n5_lepm; log exp/v5_nn.log |
| E0069 | 0.32149 | log elec per meter | 20.2 | legacy run n4_lepm; log exp/v4_nn.log |
| E0118 | 0.32173 | elec per area | 12.0 | legacy run n6_epa; log exp/v6_nn_elec.log |
| E0124 | 0.32179 | elec per area | 11.5 | legacy run n6b_epa; log exp/v6b_nn_elec.log |


### mlp_multi / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0175 | 0.69025 | {"default": {"type": "identity"}} | 1.6 |  |


### mlp_multi / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0175 | 0.32759 | {"default": {"type": "identity"}} | 1.6 |  |


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


### other / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| EM0001 | 0.65977 | gas per meter (bld_dim_12) | 50.0 | member xk_gpm |


### ridge / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0170 | 0.79288 | {"default": {"type": "identity"}} | 2.2 |  |
| E0180 | 0.79515 | {"default": {"type": "identity"}} | 2.2 |  |


### ridge / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0170 | 0.35558 | {"default": {"type": "identity"}} | 2.2 |  |
| E0180 | 0.35649 | {"default": {"type": "identity"}} | 2.2 |  |


### tabm_mini_plr / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0153 | 0.66136 | gas per meter (bld_dim_12) | 116.1 | legacy run t1_gpm; log exp/v11_nn.log |


### tabm_mini_plr / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0164 | 0.31938 | elec per meter | 265.2 | legacy run t1_epm_wM; log exp/v11_nn_d.log |
| E0149 | 0.31992 | log elec per meter | 176.4 | legacy run t1_lepm; log exp/v11_nn.log; member of blend v15nc |


### xgb / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| EM0002 | 0.65741 | gas per meter (bld_dim_12), count:poisson |  | member xp_gpm; v17 selection: blend total -0.000035 with all four, not adopted (D007); log logs/v17_full_xp_gpm.log |
| EM0005 | 0.65985 | gas per bld_dim_03, count:poisson |  | member xp_gpgea; v17 selection: blend total -0.000035 with all four, not adopted (D007); log logs/v17_full_xp_gpgea.log |
| EM0003 | 0.66008 | gas per area (bld_dim_06), count:poisson |  | member xp_eui; v17 selection: blend total -0.000035 with all four, not adopted (D007); log logs/v17_full_xp_eui.log |
| E0264 | 0.66054 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_power": 1}} | 35.0 |  |
| E0280 | 0.66199 | gas per meter (bld_dim_12), weight M^2/A | 19.1 | rejected: gain ~1/5 of FB07, EUI not significant (SE 0.00048); expected blend gain ~-0.00003 |
| E0191 | 0.66229 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}} | 21.3 |  |
| E0279 | 0.66243 | gas per meter (bld_dim_12), weight M^2/A | 9.7 | A/B reference |
| E0265 | 0.66307 | {"default": {"exposure_col": "bld_dim_06", "type": "ratio", "weight_power": 1}} | 36.9 |  |
| E0190 | 0.66444 | {"default": {"exposure_col": "bld_dim_12", "type": "ratio", "weight_col": "bld_dim_06", "weight_col_power": -1, "weight_power": 2}} | 5.7 |  |
| E0173 | 0.68645 | {"default": {"type": "identity"}} | 35.6 |  |


### xgb / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| EM0004 | 0.31977 | elec per meter (bld_dim_12), count:poisson |  | member xp_epm; v17 selection: blend total -0.000035 with all four, not adopted (D007); log logs/v17_full_xp_epm.log |
| E0173 | 0.33009 | {"default": {"type": "identity"}} | 35.6 |  |


### xgboost / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0162 | 0.66060 | gas per meter (bld_dim_12) | 84.6 | legacy run xs_gpm; log exp/v13_xs.log |
| E0132 | 0.66094 | gas per meter (bld_dim_12) | 75.9 | legacy run xa_gpm; log exp/v10_agg.log; member of blend v15nc |
| E0126 | 0.66131 | gas per meter (bld_dim_12) | 18.8 | legacy run x8_gpm; log exp/v8_xgb.log; member of blend v15nc |
| E0136 | 0.66294 | gas per bld_dim_03 | 38.2 | legacy run xa_gpgea; log exp/v10_agg_b.log; member of blend v15nc |
| E0091 | 0.66332 | gas per meter (bld_dim_12) | 10.9 | legacy run x4_gpm; log exp/v4_xgb.log |
| E0127 | 0.66351 | gas per bld_dim_03 | 15.7 | legacy run x8_gpgea; log exp/v8_xgb.log |
| E0139 | 0.66465 | log gas per meter | 36.1 | legacy run xa_lgpm; log exp/v10_agg_b.log; member of blend v15nc |
| E0095 | 0.66513 | gas per bld_dim_03 | 11.2 | legacy run x4_gpgea; log exp/v4_xgb.log |
| E0128 | 0.66525 | log gas per meter | 15.9 | legacy run x8_lgpm; log exp/v8_xgb.log; member of blend v15nc |
| E0074 | 0.66536 | EUI | 25.7 | legacy run x4_eui_wA; log exp/v4_xgb.log |


### xgboost / t2 elec

| exp_id | nrmse_t2 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0081 | 0.32197 | elec per meter | 15.0 | legacy run x4_epm_wM; log exp/v4_xgb.log; member of blend v15nc |
| E0143 | 0.32199 | elec per meter | 25.9 | legacy run xa_epm_wM; log exp/v10_agg_b.log |
| E0066 | 0.32208 | elec per meter | 15.9 | legacy run xgb_epm_wM_tew_k10; log exp/k10.log; member of blend v15nc |
| E0165 | 0.32212 | elec per meter | 31.5 | legacy run xs_epm_wM; log exp/v13_xs.log |
| E0142 | 0.32268 | log elec per area | 24.6 | legacy run xa_lepa; log exp/v10_agg_b.log; member of blend v15nc |
| E0058 | 0.32276 | elec per meter | 7.2 | legacy run xgb_epm_wM_tewc; log exp/cell1.log |
| E0089 | 0.32291 | elec per area | 6.8 | legacy run x4_epa; log exp/v4_xgb.log; member of blend v15nc |
| E0023 | 0.32291 | elec per meter | 1.9 | legacy run xgb_epm_wM_te; log exp/zoo3.log |
| E0075 | 0.32293 | log elec per area | 16.6 | legacy run x4_lepa; log exp/v4_xgb.log; member of blend v15nc |
| E0024 | 0.32294 | elec per meter | 2.1 | legacy run xgb_epm_wM_tew; log exp/zoo3.log |


### xgboost_linear / t3 EUI

| exp_id | nrmse_t3 | target_transform | runtime_min | notes |
|---|---|---|---|---|
| E0070 | 0.66815 | EUI | 64.0 | legacy run xgbl_eui_wA_tew; log exp/zoo6.log |


## 20 eksperimen terakhir

| exp_id | status | model_family | nrmse_mean | nrmse_t1 | nrmse_t2 | nrmse_t3 | hypothesis |
|---|---|---|---|---|---|---|---|
| E0264 | done | xgb |  | 0.25085 |  | 0.66054 | Poisson XGB pada gas per meter (bobot M, offset log M), mcw setara 50 baris; vs E0190 (MSE rasio bobot M^2/A) |
| E0265 | done | xgb |  | 0.25043 |  | 0.66307 | Poisson XGB pada EUI (laju per luas, bobot A = Poisson gas dengan offset log A), mcw setara 50 baris; vs E0190 |
| ENS0018 | done | ensemble | 0.24960 | 0.24524 | 0.31608 | 0.65101 | NNLS blend v16 (34 anggota, FB07 kNN-target; exp/analysis/select_v16.py) = Final A (D006) |
| EM0002 | done | xgb |  | 0.24899 |  | 0.65741 | XGB Poisson gas per meter (bobot M), input v16 + kNN-target, 10-fold |
| EM0003 | done | xgb |  | 0.24873 |  | 0.66008 | XGB Poisson EUI (laju per luas, bobot A), input v16 + kNN-target, 10-fold |
| EM0004 | done | xgb |  |  | 0.31977 |  | XGB Poisson listrik per meter (bobot M), input v16 + kNN-target, 10-fold |
| EM0005 | done | xgb |  | 0.24911 |  | 0.65985 | XGB Poisson gas per luas selubung bld_dim_03, input v16 + kNN-target, 10-fold |
| E0271 | done | lgb |  | 0.2572729456806092 |  | 0.6720985399324281 | CV_SECONDARY check: LGB gpm tanpa TE spasial, skema primary |
| E0272 | done | lgb |  |  | 0.3262256196151608 |  | CV_SECONDARY check: LGB epm tanpa TE spasial, skema primary |
| E0273 | done | lgb |  | 0.25617055067299915 |  | 0.6699013376971004 | CV_SECONDARY check: LGB gpm dengan TE spasial, skema primary |
| E0279 | done | xgb |  | 0.25198 |  | 0.66243 | FB07b acuan: XGB gpm x4 5-fold, input v16 (agg + TE + sel + FB07 paralel) |
| E0280 | done | xgb |  | 0.25167 |  | 0.66199 | FB07b: + kNN-target di ruang era / bangunan+sensus / ukuran, W=10 dan W=1000 |
| E0274 | done | lgb |  |  | 0.32545917304677086 |  | CV_SECONDARY check: LGB epm dengan TE spasial, skema primary |
| E0275 | done | lgb |  | 0.2708063678657521 |  | 0.6923146162595597 | CV_SECONDARY check: LGB gpm tanpa TE spasial, skema secondary |
| E0276 | done | lgb |  |  | 0.330181558948791 |  | CV_SECONDARY check: LGB epm tanpa TE spasial, skema secondary |
| E0277 | done | lgb |  | 0.36374811605371604 |  | 0.8330538274343691 | CV_SECONDARY check: LGB gpm dengan TE spasial, skema secondary |
| E0278 | done | lgb |  |  | 0.3473564584844149 |  | CV_SECONDARY check: LGB epm dengan TE spasial, skema secondary |
| PP0005 | done | lgb |  |  |  |  | adoption v3: census-kNN pooled q, unseen-mix training weights, EPC subzone sums, model averaging (unseen-subzone model) |
| ENS0019 | done | ensemble |  | 0.244817 | 0.315877 | 0.649839 | varying-coefficient NNLS blend: member weights multilinear in ranks of floor area, meters and global blend prediction (A2xM2xp2) |
| PP0006 | done | postprocess |  | 0.24514 | 0.31462 | 0.64859 | re-select the similarity-kNN residual-correction configs for the VC blend (greedy over 39 configs on the holdout) |


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
| 2026-10-15 16:50 | submissions/sub_v15ncrkd.csv |  | 0.24954 |  submitted by user 16 Okt |
| 2026-10-16 11:25 | submissions/BEST_CURRENT.csv | 0.25002 pre-corr; holdout corr -0.0009 | 0.24954 | BEST_CURRENT submitted by user 16 Okt |
| 2026-10-16 20:27 | submissions/sub_v16dev.csv | blend honest 0.242467 (gas+eui+elec)/5; holdout chain -0.00054 vs v15 | 0.24943 | kandidat v16, ekspektasi LB ~0.2490 submitted by user 17 Okt |
| 2026-10-18 02:51 | submissions/sub_v18dev.csv | 0.24940 (nested; holdout chain -0.00006 vs v16) | 0.24936 | submitted by user 18 Okt |
| 2026-10-18 11:42 | submissions/sub_v19dev.csv | 0.24937 (nested; holdout chain -0.00002 vs v18) | 0.24934 | submitted by user 18 Okt |
| 2026-10-19 22:51 | submissions/sub_v19vce.csv | 0.64859 | 0.24923 | holdout chain EUI 0.64859 vs 0.64878 (honest split-half -0.000106); gas/elec/clean/adoption byte-identical to sub_v19vc |
| 2026-10-19 22:53 | submissions/BEST_CURRENT.csv | 0.64859 | 0.24923 | BEST_CURRENT v19vce (LB 0.24923) |
