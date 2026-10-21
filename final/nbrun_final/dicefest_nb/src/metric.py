"""Official competition metric (verbatim) plus a per-target helper."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

TARGET_COLS = [
    'target_macro_gas_kwh',
    'target_macro_elec_kwh',
    'target_eui_gas_kwh_per_sqm',
    'target_clean_building_ratio',
    'target_electric_heating_adoption'
]

OFFICIAL_SIGMAS = {
    'target_macro_gas_kwh': 151750.04,
    'target_macro_elec_kwh': 41725.37,
    'target_eui_gas_kwh_per_sqm': 40.65,
    'target_clean_building_ratio': 12.49,
    'target_electric_heating_adoption': 7.18
}


def compute_column_wise_nrmse(y_true, y_pred, sigmas=OFFICIAL_SIGMAS):
    y_true_vals = y_true[TARGET_COLS].values if isinstance(y_true, pd.DataFrame) else np.array(y_true)
    y_pred_vals = y_pred[TARGET_COLS].values if isinstance(y_pred, pd.DataFrame) else np.array(y_pred)
    nrmse_list = []
    for idx, col in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_true_vals[:, idx], y_pred_vals[:, idx]))
        nrmse_list.append(rmse / sigmas[col])
    return float(np.mean(nrmse_list))


def nrmse_per_target(y_true, y_pred, targets=TARGET_COLS, sigmas=OFFICIAL_SIGMAS):
    """dict target -> RMSE / sigma for any subset of targets; inputs are DataFrames or dicts of arrays."""
    return {t: float(np.sqrt(np.mean((np.asarray(y_true[t], float) - np.asarray(y_pred[t], float)) ** 2)) / sigmas[t])
            for t in targets}
