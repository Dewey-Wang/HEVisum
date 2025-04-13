import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import math

def plot_cell_expression_all_slides(
    cell_short_name='C17',
    cell_prefix='zscore_log2_filtered_',
    image_path="./dataset/elucidata_ai_challenge_data.h5",
    spot_path="./dataset/processed_train_spots.h5",
    group_name="log2_Train"
):
    full_col_name = cell_prefix + cell_short_name

    # ========= 讀取影像和 spots ==========
    with h5py.File(image_path, "r") as img_h5, h5py.File(spot_path, "r") as spot_h5:
        train_images = img_h5["images/Train"]
        train_spots = spot_h5[f"spots/{group_name}"]
        slide_ids = list(train_spots.keys())

        n = len(slide_ids)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        for i, slide_id in enumerate(slide_ids):
            df = pd.DataFrame(np.array(train_spots[slide_id]))
            image = np.array(train_images[slide_id])
            if full_col_name not in df.columns:
                print(f"⚠️ {full_col_name} not in {slide_id}, skipping")
                continue
            x = df["x"].values
            y = df["y"].values
            expr = df[full_col_name].values

            ax = axes[i]
            ax.imshow(image)
            sc = ax.scatter(x, y, c=expr, cmap="plasma", s=10, alpha=0.8)
            ax.set_title(f"{slide_id}")
            ax.axis("off")

        # remove unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.2, 0.015, 0.6])
        fig.colorbar(sc, cax=cbar_ax, label='Expression')
        plt.suptitle(f"🔬 Spatial Distribution of {cell_short_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.93, 1])
        plt.show()



def plot_all_celltypes_on_slide(
    slide_id,
    cell_prefix="zscore_log2_filtered_",
    image_path="./dataset/elucidata_ai_challenge_data.h5",
    spot_path="./dataset/processed_train_spots.h5",
    group_name="log2_Train"
):
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import pandas as pd

    # 讀取影像
    with h5py.File(image_path, "r") as h5file:
        image = np.array(h5file["images/Train"][slide_id])

    # 讀取 spot 資料
    with h5py.File(spot_path, "r") as f:
        train_spots = f[f"spots/{group_name}"]
        df = pd.DataFrame(np.array(train_spots[slide_id]))

    x = df["x"].values
    y = df["y"].values

    # 抓出所有要畫的欄位
    cell_types = [col for col in df.columns if col.startswith(cell_prefix)]

    # 建立 subplot 畫布
    n_cols = 5
    n_rows = int(np.ceil(len(cell_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for idx, cell in enumerate(cell_types):
        ax = axes[idx]
        expr = df[cell].values
        ax.imshow(image, aspect='auto')
        sc = ax.scatter(x, y, c=expr, cmap='plasma', s=10, alpha=0.8)
        ax.set_title(cell)
        ax.axis('off')

    # 把多餘的空圖清掉
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])  # colorbar 的位置
    fig.colorbar(sc, cax=cbar_ax, label='Expression')

    plt.suptitle(f"{slide_id} - All Cell Type Expression Maps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.show()


def plot_cell_expression_on_slide(
    slide_id,
    cell_type,
    cell_prefix="zscore_log2_filtered_",  # or zscore_boxcox_filtered_
    image_path="./dataset/elucidata_ai_challenge_data.h5",
    spot_path="./dataset/processed_train_spots.h5",
    group_name="log2_Train"
):
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # 取得影像
    with h5py.File(image_path, "r") as h5file:
        image = np.array(h5file["images/Train"][slide_id])

    # 取得 spot table
    with h5py.File(spot_path, "r") as f:
        df = pd.DataFrame(np.array(f[f"spots/{group_name}"][slide_id]))

    x = df["x"].values
    y = df["y"].values
    full_col_name = cell_type if cell_type.startswith(cell_prefix) else cell_prefix + cell_type

    if full_col_name not in df.columns:
        raise ValueError(f"Cell type '{full_col_name}' 不存在於該 slide 的資料中！")

    expr = df[full_col_name].values

    # 畫圖
    plt.figure(figsize=(8, 8))
    plt.imshow(image, aspect="auto")
    sc = plt.scatter(x, y, c=expr, cmap='plasma', s=15, alpha=0.8)
    plt.colorbar(sc, label=f"{full_col_name}")
    plt.title(f"{slide_id} - {full_col_name} Expression Map")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
