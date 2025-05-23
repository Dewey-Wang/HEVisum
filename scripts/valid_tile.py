import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def to_uint8(img):
    """Convert float [0,1] or uint8 to uint8 RGB."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img/255.0, 0, 1)
        img = (img * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def visualize_tile_vs_reconstructed(tile, subtiles, slide_idx, position, title="Tile vs Reconstructed"):
    """
    Display two images side by side:
      - Left: the original `tile` (78×78×3)
      - Right: the 3×3 mosaic reconstructed from `subtiles` (9×(26×26×3))
    Also shows slide_idx and position in the figure title.
    """
    tile_u8 = to_uint8(tile)
    # subtiles: shape (9, 26, 26, 3)
    sH, sW = subtiles.shape[1:3]
    # reconstruct mosaic: 3 rows of 3 subtiles
    rows = []
    for i in range(3):
        row = np.hstack([to_uint8(subtiles[i*3 + j]) for j in range(3)])
        rows.append(row)
    mosaic = np.vstack(rows)  # shape (78, 78, 3)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(tile_u8)
    axes[0].set_title("Original Tile")
    axes[0].axis("off")

    axes[1].imshow(mosaic)
    axes[1].set_title("Reconstructed from Subtiles")
    axes[1].axis("off")

    fig.suptitle(f"{title}\nSlide: {slide_idx} | Position: {position}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.show()

def visualize_random_sample(data_dir="dataset/try/train_data"):
    pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
    if not pt_files:
        print("❌ No .pt files found in", data_dir)
        return

    chosen = random.choice(pt_files)
    path = os.path.join(data_dir, chosen)
    data = torch.load(path)

    tile      = data["tile"]       # (78,78,3)
    subtiles  = data["subtiles"]   # (9,26,26,3)
    slide_idx = data.get("slide_idx", "N/A")
    position  = data.get("position", "N/A")

    print(f"🧪 Visualizing: {chosen}")
    visualize_tile_vs_reconstructed(tile, subtiles, slide_idx, position, title=chosen)
# Example usage:
