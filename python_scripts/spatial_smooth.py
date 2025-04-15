from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd

def gaussian_spatial_smoothing(df, cell_type_cols, radius=80, sigma=None):
    coords = df[["x", "y"]].values
    tree = KDTree(coords)
    sigma = sigma or (radius / 2)

    smoothed = []

    for i in range(len(df)):
        # 找鄰居及距離
        idxs, dists = tree.query_radius([coords[i]], r=radius, return_distance=True)
        idxs = idxs[0]
        dists = dists[0]
        
        if len(idxs) == 0:
            smoothed.append(df.iloc[i][cell_type_cols].values)  # 沒鄰居就原值
            continue
        
        # 計算權重（高斯）
        weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
        weights = weights / (weights.sum() + 1e-8)  # normalize
        
        # 加權平均
        neighbor_vals = df.iloc[idxs][cell_type_cols].values  # (N, C)
        weighted_vals = np.average(neighbor_vals, axis=0, weights=weights)
        smoothed.append(weighted_vals)
    
    smoothed_array = np.vstack(smoothed)
    smoothed_df = df.copy()
    smoothed_df[cell_type_cols] = smoothed_array
    return smoothed_df

import matplotlib.pyplot as plt
import numpy as np
import h5py

def plot_cell_expression_comparison(
    slide_id,
    cell_type,
    original_df,
    radius,
    smoothed_df,
    h5_path="./dataset/elucidata_ai_challenge_data.h5",
    cmap="plasma"
):
    """
    比較原始 vs 平滑後的細胞類型表達分佈（統一顏色區間），並印出 Spearman/Pearson。
    """
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    with h5py.File(h5_path, "r") as f:
        image = np.array(f["images/Train"][slide_id])

    x = original_df["x"].values
    y = original_df["y"].values
    z_orig = original_df[cell_type].values
    z_smooth = smoothed_df[cell_type].values

    # 統一顏色範圍
    vmin = min(z_orig.min(), z_smooth.min())
    vmax = max(z_orig.max(), z_smooth.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 計算 correlation
    spearman_corr = spearmanr(z_orig, z_smooth)
    pearson_corr = pearsonr(z_orig, z_smooth)

    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    sc1 = axes[0].scatter(x, y, c=z_orig, cmap=cmap, s=15, norm=norm)
    axes[0].set_title(f"{slide_id} - Original Expression")
    axes[0].axis("off")
    plt.colorbar(sc1, ax=axes[0], label=cell_type)

    axes[1].imshow(image)
    sc2 = axes[1].scatter(x, y, c=z_smooth, cmap=cmap, s=15, norm=norm)
    axes[1].set_title(f"{slide_id} - Smoothed Expression (r={radius})")
    axes[1].axis("off")
    plt.colorbar(sc2, ax=axes[1], label=cell_type)

    plt.suptitle(f"Cell type {cell_type} - Expression comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 顯示 correlation 結果
    print(f"▶️ Spearman ρ = {spearman_corr.statistic:.4f}, p = {spearman_corr.pvalue:.4g}")
    print(f"▶️ Pearson  r = {pearson_corr.statistic:.4f}, p = {pearson_corr.pvalue:.4g}")


import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def smooth_all_slides_and_save(train_spot_tables, cell_type_cols, output_path, radius=80):
    with h5py.File(output_path, "w") as f:
        group = f.create_group("spots/log2_Train")
        
        for slide_id, df in tqdm(train_spot_tables.items(), desc="Processing slides"):
            smoothed_df = gaussian_spatial_smoothing(df, cell_type_cols, radius=radius)
            save_df = smoothed_df[["x", "y"] + cell_type_cols]
            rec_array = save_df.to_records(index=False)
            group.create_dataset(slide_id, data=rec_array)
        
    print(f"✅ 所有 slide 都已平滑並存成 {output_path}，格式與 processed_train_spots.h5 相同！")