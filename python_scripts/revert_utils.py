# revert_utils.py

import json
import numpy as np
from scipy.special import inv_boxcox

# ---------------------------
# 🔹 通用工具
# ---------------------------

def load_json_params(json_path):
    """通用的 JSON 載入器（可載入 zscore 或 boxcox_zscore）"""
    with open(json_path, "r") as f:
        return json.load(f)


# ---------------------------
# 🔹 Z-score + log2 還原
# ---------------------------

def revert_zscore_to_log2(zscore_val, mean, std):
    return zscore_val * std + mean

def revert_log2(log2_val, add_constant=1.0):
    return np.power(2, log2_val) - add_constant

def revert_log2_predictions(predictions, zscore_params, add_constant=1.0):
    predictions = np.asarray(predictions)
    restored = np.zeros_like(predictions)

    cell_types = sorted(zscore_params.keys(), key=lambda x: int(x[1:]))
    for i, cell in enumerate(cell_types):
        mean = zscore_params[cell]["mean"]
        std = zscore_params[cell]["std"]

        log2_val = revert_zscore_to_log2(predictions[:, i], mean, std)
        restored[:, i] = revert_log2(log2_val, add_constant=add_constant)

    return restored


# ---------------------------
# 🔹 Z-score + Box-Cox 還原
# ---------------------------

def revert_zscore_to_boxcox(zscore_val, mean, std):
    return zscore_val * std + mean

def revert_boxcox(box_val, lam, add_constant):
    original = inv_boxcox(box_val, lam)
    return original - add_constant

def revert_boxcox_predictions(predictions, param_dict, eps=1e-6, fillna_strategy='median'):
    predictions = np.asarray(predictions)
    restored = np.zeros_like(predictions)

    cell_types = sorted(param_dict.keys(), key=lambda x: int(x[1:]))
    for i, cell in enumerate(cell_types):
        params = param_dict[cell]
        mean = params["mean"]
        std = params["std"]
        lam = params["lambda"]
        add_const = params["add_constant"]

        box_val = revert_zscore_to_boxcox(predictions[:, i], mean, std)
        box_val = np.maximum(box_val, eps)

        with np.errstate(invalid='ignore'):
            orig_val = revert_boxcox(box_val, lam, add_const)

        # ⚠️ 修正 NaN
        nan_mask = np.isnan(orig_val)
        if np.any(nan_mask):
            if fillna_strategy == "zero":
                orig_val[nan_mask] = 0.0
            elif fillna_strategy == "median":
                orig_val[nan_mask] = np.nanmedian(orig_val)
            elif fillna_strategy == "mean":
                orig_val[nan_mask] = np.nanmean(orig_val)
            else:
                raise ValueError(f"未知 fillna_strategy: {fillna_strategy}")
            print(f"⚠️ {cell}: {np.sum(nan_mask)} NaNs filled with {fillna_strategy}")

        restored[:, i] = orig_val

    return restored


# ---------------------------
# ✅ 使用範例 (可刪除或改成 test)
# ---------------------------

if __name__ == "__main__":
    dummy_preds = np.random.randn(10, 35)

    print("🔁 測試 revert_log2_predictions")
    zscore_params = load_json_params("./dataset/zscore_params.json")
    orig_vals_log2 = revert_log2_predictions(dummy_preds, zscore_params)
    print(orig_vals_log2[:2])

    print("🔁 測試 revert_boxcox_predictions")
    boxcox_params = load_json_params("./dataset/boxcox_zscore_params.json")
    orig_vals_boxcox = revert_boxcox_predictions(dummy_preds, boxcox_params)
    print(orig_vals_boxcox[:2])
