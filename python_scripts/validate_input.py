import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float or uint8 image to RGB uint8 format."""
    if img.dtype != np.uint8:
        # Normalize float images in [0,1] or [0,255]
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img / 255.0, 0, 1)
        img = (img * 255).astype(np.uint8)
    # Drop alpha channel if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


def draw_border(img: np.ndarray, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw a border around the image."""
    h, w = img.shape[:2]
    return cv2.rectangle(img.copy(), (0, 0), (w - 1, h - 1), color, thickness)


def visualize_tiles_and_subtiles(
    tile: np.ndarray,
    subtiles: list,
    neighbors: list = None,
    title_prefix: str = "Tile",
    subtile_grid_size: int = 2
):
    """
    Display:
    - If neighbors provided, show a 3x3 grid of center tile + up to 8 neighbors.
    - Show the center tile's subtiles in a grid (2x2 or 3x3).
    """
    # Prepare center tile
    tile_img = to_uint8(tile)
    tile_img = draw_border(tile_img)

    # Prepare neighbors grid if available
    show_neighbors = neighbors is not None and len(neighbors) > 0
    if show_neighbors:
        neigh_imgs = [draw_border(to_uint8(n)) for n in neighbors]
    
    # Prepare subtiles
    sub_imgs = [draw_border(to_uint8(s)) for s in subtiles]

    # Plotting
    cols = 1 + int(show_neighbors)  # 1 col for subtiles, +1 if neighbors
    plt.figure(figsize=(6 * cols, 6))

    # Plot neighbors + center
    if show_neighbors:
        H, W = tile_img.shape[:2]
        grid3x3 = np.zeros((3 * H, 3 * W, 3), dtype=np.uint8)
        idx = 0
        for i in range(3):
            for j in range(3):
                y1, y2 = i * H, (i + 1) * H
                x1, x2 = j * W, (j + 1) * W
                if i == 1 and j == 1:
                    patch = tile_img
                else:
                    # If fewer than 8 neighbors, fill with blank
                    if idx < len(neigh_imgs):
                        patch = neigh_imgs[idx]
                    else:
                        patch = np.zeros((H, W, 3), dtype=np.uint8)
                    idx += 1
                grid3x3[y1:y2, x1:x2] = patch

        plt.subplot(1, cols, 1)
        plt.imshow(grid3x3)
        plt.title(f"{title_prefix} - Neighbors + Center")
        plt.axis("off")

    # Plot subtiles
    sH, sW = sub_imgs[0].shape[:2]
    grid_sub = np.zeros((subtile_grid_size * sH, subtile_grid_size * sW, 3), dtype=np.uint8)
    for k, img in enumerate(sub_imgs):
        i = k // subtile_grid_size
        j = k % subtile_grid_size
        y1, y2 = i * sH, (i + 1) * sH
        x1, x2 = j * sW, (j + 1) * sW
        grid_sub[y1:y2, x1:x2] = img

    plt.subplot(1, cols, cols)
    plt.imshow(grid_sub)
    plt.title(f"{title_prefix} - {subtile_grid_size}×{subtile_grid_size} Subtiles")  
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_random_from_grouped_data(grouped_data: dict, subtile_grid_size: int = 2):
    """
    Randomly pick one sample from grouped_data and visualize its tile and subtiles.

    grouped_data: {
        'tile': list or array of tile images,
        'subtiles': list of lists/arrays of subtile images,
        ... other keys ...
    }
    """
    # Ensure keys exist
    if 'tile' not in grouped_data or 'subtiles' not in grouped_data:
        raise KeyError("grouped_data must contain 'tile' and 'subtiles' keys.")

    # Number of samples
    n = len(grouped_data['tile'])
    if n == 0:
        print("No samples in grouped_data.")
        return

    # Choose random index
    idx = random.randint(0, n - 1)

    tile = grouped_data['tile'][idx]
    subtiles = grouped_data['subtiles'][idx]

    # Optional: if 'neighbors' in grouped_data, pass it
    neighbors = grouped_data.get('neighbors', None)
    title = f"Sample {idx}"  

    visualize_tiles_and_subtiles(
        tile=tile,
        subtiles=subtiles,
        neighbors=neighbors,
        title_prefix=title,
        subtile_grid_size=subtile_grid_size
    )
