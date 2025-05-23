def split_into_subtiles(tile, grid_size=3):
    """
    將 tile 切成 grid_size x grid_size 的 subtiles
    """
    H, W, C = tile.shape
    assert H % grid_size == 0 and W % grid_size == 0
    h_step, w_step = H // grid_size, W // grid_size

    subtiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step
            sub_tile = tile[y1:y2, x1:x2, :]
            subtiles.append(sub_tile)
    return subtiles  # 會回傳 9 個 (H/3, W/3, C)



def extract_tile_fixed(image, x, y, tile_size=336, pad_mode='edge'):
    """
    擷取中心為 (x, y) 的 tile，若超出邊界則 padding，並回傳 padding 過的位置（若有）。
    
    Args:
        image: 原始影像 (H, W, C)
        x, y: 中心座標
        tile_size: 擷取的 tile 大小
        pad_mode: np.pad 使用的模式（預設為 'edge'）

    Returns:
        tile: 擷取出來的 tile (tile_size, tile_size, C)
        padded_coords: 若有 padding 則為 [x, y]，否則為 None
    """
    H, W, C = image.shape
    half = tile_size // 2

    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half

    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)

    padded = any([pad_left, pad_right, pad_top, pad_bottom])
    if padded:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=pad_mode
        )
        # 調整座標
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    tile = image[y1:y2, x1:x2, :]
    return (tile, [x, y]) if padded else (tile, None)



def get_spots_in_tile(df, center_x, center_y, tile_size):
    half = tile_size // 2
    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    df_in_tile = df[
        (df['x'] >= x_min) & (df['x'] <= x_max) &
        (df['y'] >= y_min) & (df['y'] <= y_max)
    ].copy()

    df_in_tile = df_in_tile[
        ~((df_in_tile['x'] == center_x) & (df_in_tile['y'] == center_y))
    ]

    return df_in_tile


import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_tile_with_spots(
    slide_image, spot_df, center_x, center_y,
    tile_size=78, stride=None, grid_size=None,
    spot_radius_px=15
):
    """
    可視化一個 tile，包含中心 spot、鄰近 spot，以及可選的 N×N 子區格線。

    Parameters:
    - slide_image: 該張切片的影像 (numpy array, H x W x 3)
    - spot_df: 包含所有落在 tile 內的 spots（已排除中心點）
    - center_x, center_y: 中心 spot 的 slide 座標
    - tile_size: 單一 tile 的大小（像素）
    - stride: tile 間距（默認等於 tile_size，表示無重疊）
    - grid_size: 幾乘幾的 tile（如果 None，就只畫單 tile）
    - spot_radius_px: 中心 spot 畫圓半徑
    """

    if stride is None:
        stride = tile_size

    if grid_size:
        full_tile_size = tile_size + stride * (grid_size - 1)
    else:
        full_tile_size = tile_size

    center_color = 'red'
    neighbor_color = 'orange'
    show_legend = True
    title = f"{len(spot_df)} neighbors in tile"

    # 擷取 tile
    def extract_tile(slide_img, x, y, total_size):
        x, y = int(x), int(y)
        half = total_size // 2
        x1, x2 = x - half, x + half
        y1, y2 = y - half, y + half
        tile = slide_img[y1:y2, x1:x2]
        return tile

    tile = extract_tile(slide_image, center_x, center_y, full_tile_size)

    # 畫圖
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tile)

    # 畫格線
    # 畫出每個 tile 的實際切割框
    if grid_size:
        for row in range(grid_size):
            for col in range(grid_size):
                x0 = col * stride
                y0 = row * stride
                rect = patches.Rectangle(
                    (x0, y0),
                    tile_size, tile_size,
                    linewidth=1,
                    edgecolor='blue',
                    facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)


    # 中心 spot（固定畫在 tile 中央）
    center_px = full_tile_size // 2
    ax.scatter(center_px, center_px, c=center_color, s=40, label='Center Spot')

    # 中心 spot 圓圈
    circle = patches.Circle(
        (center_px, center_px),
        spot_radius_px,
        linewidth=1.5,
        edgecolor='yellow',
        facecolor='none'
    )
    ax.add_patch(circle)

    # 鄰居 spots（轉成 tile 內相對座標）
    for _, neighbor in spot_df.iterrows():
        dx = int(neighbor['x']) - center_x
        dy = int(neighbor['y']) - center_y
        tile_x = center_px + dx
        tile_y = center_px + dy
        ax.scatter(tile_x, tile_y, c=neighbor_color, s=20)

    ax.axis('off')
    if show_legend:
        ax.legend(loc='upper right')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
