# revert_predict_data.py

import json
import numpy as np

# 固定的 JSON 參數檔案路徑（請依需求修改）
Z_SCORE_PARAMS_PATH = "/Users/deweywang/Desktop/GitHub/HEVisum/data preprocessing/spot data cleaning/zscore_params.json"


def load_zscore_params(file_path=Z_SCORE_PARAMS_PATH):
    """
    從指定路徑讀取 Z-score 參數 JSON 檔案，並返回參數字典。
    
    JSON 格式示例：
    {
        "C1": {"mean": 0.2755919685587203, "std": 0.46319316072500044},
        "C2": {"mean": 0.12125104833703637, "std": 0.1044255861975191},
        ...
        "C35": {"mean": 0.05399294893765085, "std": 0.05767095842989666}
    }
    
    回傳:
        params: dict，key 為 cell type (例如 "C1")，value 為包含 mean 與 std 的字典
    """
    with open(file_path, "r") as f:
        params = json.load(f)
    return params


def revert_zscore(zscore_value, mean, std):
    """
    反轉 Z-score 標準化
    將模型預測的 Z-score 乘上標準差並加上均值，還原成 log₂ 轉換後的值。
    
    參數:
        zscore_value: 可為標量、NumPy 陣列或 PyTorch Tensor 的 Z-score 值
        mean: 該 cell type 的平均值（在 log₂ 轉換後的資料上）
        std: 該 cell type 的標準差
    回傳:
        log2_value: 還原後的 log₂ 表達值
    """
    return zscore_value * std + mean


def revert_log2(log2_value, add_constant=1):
    """
    反轉 Log₂ 轉換
    利用公式 x = 2^(log2_value) - add_constant 還原原始表達數值。
    預設加數為 1，即原始轉換為 log2(x+1)。
    
    參數:
        log2_value: 經過 log₂ 轉換後的數值（可為數值或 NumPy 陣列）
        add_constant: 在 log₂ 轉換時所加的常數，預設為 1
    回傳:
        original_value: 還原後的原始表達數值
    """
    return np.power(2, log2_value) - add_constant


def revert_prediction(zscore_value, zscore_params, add_constant=1):
    """
    反轉單一 cell type 的預測結果
    先利用已保存的 mean 與 std 參數反轉 Z-score，
    再反轉 log₂ 轉換，最終還原原始表達數值。
    
    參數:
        zscore_value: 模型預測的 Z-score 值（數值或陣列）
        zscore_params: 包含該 cell type 'mean' 與 'std' 的字典
        add_constant: log₂ 轉換中所用的常數，預設為 1
    回傳:
        original_value: 還原後的原始表達數值
    """
    log2_value = revert_zscore(zscore_value, zscore_params["mean"], zscore_params["std"])
    original_value = revert_log2(log2_value, add_constant=add_constant)
    return original_value


def revert_prediction_array(predictions, zscore_params=None, add_constant=1):
    """
    反轉模型預測結果陣列
    假設預測的 predictions 為 NumPy 陣列，形狀為 (n_samples, n_celltypes)，
    且 cell type 的順序依照 "C1", "C2", …, "C35"。
    
    參數:
        predictions: NumPy 陣列，模型預測的 Z-score 值，形狀為 (n_samples, n_celltypes)
        zscore_params: 若為 None，則從固定路徑讀取參數；否則應為字典，
                       格式為 { "C1": {"mean":..., "std":...}, ... }。
        add_constant: log₂ 轉換時所使用的常數，預設為 1
    回傳:
        original_predictions: NumPy 陣列，還原後的原始表達數值，形狀與 predictions 相同
    """
    # 若未提供 zscore_params，從預設路徑載入
    if zscore_params is None:
        zscore_params = load_zscore_params()
    
    # 確認預測結果為 NumPy 陣列
    predictions = np.asarray(predictions)
    n_samples, n_celltypes = predictions.shape
    
    # 初始化還原結果陣列
    original_predictions = np.zeros_like(predictions, dtype=np.float32)
    
    # 為確保依照 cell type 的自然順序 "C1", "C2", ..., "C35"
    # 我們以 key 中數字排序
    cell_types = sorted(zscore_params.keys(), key=lambda x: int(x[1:]))  # 假設 key 形式為 "C1", "C2", ...
    
    # 對每個 cell type 欄位分別還原
    for i, cell in enumerate(cell_types):
        mean_val = zscore_params[cell]["mean"]
        std_val = zscore_params[cell]["std"]
        # 反轉 Z-score： predictions[:, i] 為該 cell 的 Z-score 值
        log2_vals = predictions[:, i] * std_val + mean_val
        # 反轉 Log2：還原成原始表達值
        original_predictions[:, i] = np.power(2, log2_vals) - add_constant

    return original_predictions


# 測試範例（可以移除或作為單元測試）
if __name__ == "__main__":
    # 假設有一個模型預測結果，形狀 (n_samples, 35)
    dummy_predictions = np.array([
        [0.29158217, 0.12338457, 0.17179191, 0.03849616, 0.01655579, 0.02754541],
        [0.25503415, 0.13337179, 0.15245820, 0.04139816, 0.01808124, 0.03039141]
    ], dtype=np.float32)
    # 為了測試，假設目前只有 6 個 cell types (C1 ~ C6)
    # 請注意若要測試全部 35 個，dummy_predictions 形狀需相符
    dummy_predictions = np.pad(dummy_predictions, ((0, 0), (0, 29)), mode='constant', constant_values=0)

    # 讀取參數（實際運行時會從指定路徑讀取）
    params = load_zscore_params()
    # 反轉預測，得到原始表達數值
    original_vals = revert_prediction_array(dummy_predictions, zscore_params=params, add_constant=1)
    
    print("預測的 Z-score 值 (部分):")
    print(dummy_predictions)
    print("\n還原後的原始表達數值 (部分):")
    print(original_vals)
