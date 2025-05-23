
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_tissue_mask(image_rgb, blur_kernel: int = 7) -> np.ndarray:
    """
    用灰階平均值作全域閾值，輸出 0/1 mask（tissue = 1）。

    Parameters
    ----------
    image_rgb   : ndarray (H, W, 3)  RGB 影像
    blur_kernel : int    Gaussian kernel 大小，必須是奇數

    Returns
    -------
    mask : ndarray (H, W)  uint8 (0 / 1)
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_uint8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(gray_uint8, (blur_kernel, blur_kernel), 0)

    # --- 以平均值為閾值 ---
    thresh_val = blur.mean()
    mask = (blur < thresh_val).astype(np.uint8)   # 亮背景 → 0；較暗組織 → 1
    return mask

def extract_tile(image, x, y, tile_size=56):
    half = tile_size // 2
    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def detect_invalid_spots_and_plot(image, x_coords, y_coords, 
                                   tile_size=56, title="Invalid Spot Detection", 
                                   return_invalid=True, ax=None):
    # Step 1: 建立 binary mask
    tissue_mask = generate_tissue_mask(image)

    # Step 2: 檢查每個 spot 是否 valid
    invalid_spots = []
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        tile = extract_tile(tissue_mask, x, y, tile_size=tile_size)
        if tile.sum() == 0:
            invalid_spots.append((i, x, y))
    print(f"🚫 Found {len(invalid_spots)} invalid spots (outside tissue) in slide {title}")

    # Step 3: 畫圖（可支援傳入 ax 作為子圖）
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.scatter(x_coords, y_coords, color='green', s=2, alpha=0.3, label='Valid Spots')
    if invalid_spots:
        invalid_x = [x for _, x, _ in invalid_spots]
        invalid_y = [y for _, _, y in invalid_spots]
        ax.scatter(invalid_x, invalid_y, color='red', s=8, alpha=0.8, label='Invalid Spots')
    ax.set_title(title)
    ax.axis('off')
    ax.legend()

    # Optional return
    if return_invalid:
        return invalid_spots
    
def get_invalid_spot_coords(invalid_list):
    return set((x, y) for _, x, y in invalid_list)