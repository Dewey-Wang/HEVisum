import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import math

def plot_cell_expression_all_slides(
    cell_short_name='C17',
    cell_prefix='zscore_log2_filtered_',
    image_path="./dataset/elucidata_ai_challenge_data.h5",
    spot_path="dataset/spots-data/version-4/gu_zscore_processed_train_spots.h5",  # 用於 Train 組的 spots與表達量
    group_name="log2_Train",                          # 用於 Train 組，Test 依然用相同欄位名稱（但 Test 的 CSV 只有 "C*"）
    dataset_type="Train",                             # 如果不輸入 submission_spot_data，僅顯示此組
    submission_spot_data=None                         # 如果有，則同時顯示 Train 與 Test
):
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import pandas as pd
    import math
    import os

    # 為 Train 組，建立 full_col_name；而對於 Test 組的 submission 資料則直接用 cell_short_name
    full_col_name = cell_prefix + cell_short_name

    # 如果 submission_spot_data 有值，則要顯示 Train 與 Test 兩組；否則只顯示指定的 dataset_type
    if submission_spot_data is not None:
        groups_to_plot = ["Train", "Test"]
    else:
        groups_to_plot = [dataset_type]

    results = []  # 儲存每個 slide 的資料

    for grp in groups_to_plot:
        # 讀取該組所有 slide id（從圖像部分）
        with h5py.File(image_path, "r") as img_h5:
            slide_ids = list(img_h5[f"images/{grp}"].keys())
        for slide in slide_ids:
            # ---------- 讀取圖像 ----------
            with h5py.File(image_path, "r") as h5file:
                img = np.array(h5file[f"images/{grp}"][slide])
            # ---------- 讀取 spots 座標 ----------
            # Train 組從 HDF5 的 spot_path 讀取；Test 組則從 image_path 同一檔案讀取
            if grp == "Train":
                with h5py.File(spot_path, "r") as sp_h5:
                    print(grp, slide)
                    spots = np.array(sp_h5[f"spots/{group_name}"][slide])
                    x = spots["x"]
                    y = spots["y"]
            else:
                with h5py.File(image_path, "r") as h5file:
                    spots = np.array(h5file[f"spots/{grp}"][slide])
                    x = spots["x"]
                    y = spots["y"]

            # ---------- 讀取表達量資料 ----------
            if grp == "Train":
                # Train 組：從 HDF5 讀取 spot 資料
                with h5py.File(spot_path, "r") as sp_h5:
                    df = pd.DataFrame(np.array(sp_h5[f"spots/{group_name}"][slide]))
                # 對 Train 組採用 full_col_name
                target_col = full_col_name
            else:
                # Test 組：如果有 submission_spot_data，就用它
                if submission_spot_data is not None:
                    if isinstance(submission_spot_data, str) and os.path.isfile(submission_spot_data):
                        df_test = pd.read_csv(submission_spot_data)
                        # 如果有 slide_id 欄位就過濾
                        if "slide_id" in df_test.columns:
                            df = df_test[df_test["slide_id"] == slide]
                        else:
                            df = df_test.copy()
                    elif isinstance(submission_spot_data, pd.DataFrame):
                        df_test = submission_spot_data.copy()
                        if "slide_id" in df_test.columns:
                            df = df_test[df_test["slide_id"] == slide]
                        else:
                            df = df_test.copy()
                    else:
                        raise ValueError("submission_spot_data 必須為 CSV 路徑或 pandas DataFrame。")
                    # 對於 Test 的 submission 資料，欄位名稱固定為 cell_short_name（如 "C17"）
                    target_col = cell_short_name
                else:
                    # Test 組但沒有 submission_spot_data，則從 HDF5 讀取
                    with h5py.File(spot_path, "r") as sp_h5:
                        df = pd.DataFrame(np.array(sp_h5[f"spots/{group_name}"][slide]))
                    target_col = full_col_name

            # ---------- 驗證表達量資料行數與座標數是否一致 ----------
            if df.shape[0] != len(x):
                print(f"⚠️ In slide {slide} ({grp}): 資料行數 {df.shape[0]} 與座標數 {len(x)} 不一致，跳過")
                continue

            results.append({
                "group": grp,
                "slide": slide,
                "image": img,
                "df": df,
                "x": x,
                "y": y,
                "target_col": target_col
            })

    if len(results) == 0:
        raise ValueError("沒有符合條件的 slide 進行繪圖。")

    # ---------- 建立整體子圖畫布 ----------
    n = len(results)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    used_axis = 0
    for res in results:
        grp = res["group"]
        slide = res["slide"]
        img = res["image"]
        df = res["df"]
        x = res["x"]
        y = res["y"]
        target_col = res["target_col"]

        if target_col not in df.columns:
            print(f"⚠️ {target_col} not in slide {slide} ({grp}), skipping")
            continue

        expr = df[target_col].values
        ax = axes[used_axis]
        used_axis += 1

        ax.imshow(img, aspect='auto')
        sc = ax.scatter(x, y, c=expr, cmap="plasma", s=10, alpha=0.8)
        ax.set_title(f"{slide} ({grp})")
        ax.axis("off")
    
    # 清除未使用的子圖
    for j in range(used_axis, len(axes)):
        axes[j].axis("off")
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.2, 0.015, 0.6])
    fig.colorbar(sc, cax=cbar_ax, label='Expression')
    plt.suptitle(f"🔬 Spatial Distribution of {cell_short_name} (All Slides)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.show()

def plot_all_celltypes_on_slide(
    slide_id="S_7",
    submission_spot_data=None,                         # CSV 路徑或 DataFrame
    spot_path_h5=None,                      # 若從 HDF5 抓 spot data（當 spot_data 為 None 時生效）
    image_path="./dataset/elucidata_ai_challenge_data.h5",
    image_group="Test",                     # 專門給 image 的 group（例如 "Test" 或 "Train"）
    spot_group="Test",                      # 專門給 spot 的 group
    cell_prefix=None                        # 例如 "zscore_log2_filtered_"，若為 None 則選取所有以 "C" 開頭的欄位
):
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import pandas as pd
    import math
    import os

    # ---------- 讀取圖像以及，若需要，讀取座標 ----------
    if spot_path_h5 is None:
        # 沒有提供 spot_data 也沒有 spot_path_h5，
        # 則直接從 image_path 中讀取 image 與 spots 座標
        with h5py.File(image_path, "r") as h5file:
            image = np.array(h5file[f"images/{image_group}"][slide_id])
            spots = np.array(h5file[f"spots/{spot_group}"][slide_id])
            x = spots["x"]
            y = spots["y"]
    else:
        # 如果有提供 spot_data 或 spot_path_h5，只讀取圖像
        with h5py.File(image_path, "r") as h5file:
            image = np.array(h5file[f"images/{image_group}"][slide_id])
        x = None
        y = None  # 座標後續從 HDF5 讀取

    # ---------- 讀取表達量資料 ----------
    if isinstance(submission_spot_data, str) and os.path.isfile(submission_spot_data):
        # 傳入的是 CSV 路徑
        df = pd.read_csv(submission_spot_data)
        if df.columns[0].lower() in ["id", "spot_id", "index"]:
            df = df.iloc[:, 1:]
    elif isinstance(submission_spot_data, pd.DataFrame):
        # 傳入的是 DataFrame
        df = submission_spot_data.copy()
    elif submission_spot_data is None and spot_path_h5:
        # 沒有提供 spot_data，但提供 HDF5 的 spot 檔
        with h5py.File(spot_path_h5, "r") as f:
            df = pd.DataFrame(np.array(f[f"spots/{spot_group}"][slide_id]))
        # 同時從 HDF5 中取得座標
        with h5py.File(spot_path_h5, "r") as f:
            spots_data = f[f"spots/{spot_group}"][slide_id]
            x = spots_data["x"]
            y = spots_data["y"]
    else:
        raise ValueError("請傳入有效的 spot_data（CSV 或 DataFrame）或指定 HDF5 的 spot_path")

    # ---------- 檢查座標與資料行數是否對應 ----------
    if x is not None and (df.shape[0] != len(x)):
        raise ValueError(f"資料行數 {df.shape[0]} 與座標數 {len(x)} 不一致！")

    # ---------- 擷取 cell type 欄位 ----------
    if cell_prefix:
        cell_types = [col for col in df.columns if col.startswith(cell_prefix)]
    else:
        cell_types = [col for col in df.columns if col.startswith("C")]

    if not cell_types:
        raise ValueError("找不到對應的 cell type 欄位！")

    # ---------- 建立子圖 ----------
    n_cols = 5
    n_rows = math.ceil(len(cell_types) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for idx, cell in enumerate(cell_types):
        ax = axes[idx]
        expr = df[cell].values

        # 顯示背景圖
        ax.imshow(image, aspect='auto')
        # 如果之前有從 HDF5 讀取座標則使用，如果 x, y 為 None，則報錯
        if x is None or y is None:
            raise ValueError("座標資料缺失，請檢查您的輸入")
        sc = ax.scatter(x, y, c=expr, cmap='plasma', s=10, alpha=0.8)
        ax.set_title(cell)
        ax.axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Expression')

    plt.suptitle(f"{slide_id} - Cell Type Expression Maps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.93, 0.97])
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
