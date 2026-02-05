import torch
import numpy as np
import random
import albumentations as A
from copy import deepcopy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# --------------------------------------------------------------------------------
# 1) Identity transform for importDataset
#    Returns sample unchanged.
def identity(sample):
    return sample

# --------------------------------------------------------------------------------
# 2) ReplayCompose pipeline for tile augmentation

# 全部統一的 Normalize
my_normalize = A.Normalize(
    mean=(0.72993416*255, 0.66401669*255, 0.79858416*255),
    std=(0.13688086*255, 0.16952383*255, 0.10068341*255)
)

# 強 augmentation (train/val)
train_val_transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.8),
    A.Rotate(limit=90, p=0.8),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    A.GaussianBlur(blur_limit=(3,7), p=0.5),
    A.GaussNoise(var_limit=(10.0,50.0), p=0.5),
    # my_normalize,
    # A.ElasticTransform(alpha=1, sigma=1, p=0.2),

])

# 測試基本 flip/rotate
test_transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.8),
    A.Rotate(limit=90, p=0.8),
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    # my_normalize
])

# 如果你還有其他 transform (e.g., val_transform)
# 可以和train共用:
val_transform = A.Compose([
    my_normalize
])


SEED = 42

# --------------------------------------------------------------------------------
# Helper: split augmented tile into subtiles
def split_into_subtiles(tile: np.ndarray, grid_size: int = 3) -> list[np.ndarray]:
    """
    Split a HxWxC numpy image into grid_size x grid_size subtiles.
    Returns a list of arrays each of shape (H/grid_size, W/grid_size, C).
    """
    H, W, C = tile.shape
    assert H % grid_size == 0 and W % grid_size == 0, \
        f"Tile size {tile.shape} not divisible by grid size {grid_size}"
    h_step, w_step = H // grid_size, W // grid_size
    subtiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step
            subtiles.append(tile[y1:y2, x1:x2, :])
    return subtiles

# --------------------------------------------------------------------------------
# 3) Pickle-able augmenter: augments only tile (numpy HWC), regenerates subtiles
class AugmentFn:
    def __init__(self, repeats: int = 1, grid_size: int = 3, transform=None):
        self.repeats = repeats
        self.grid_size = grid_size
        self.transform = transform

    def __call__(self, sample: dict, base_idx: int, aug_idx: int) -> dict:
        seed = SEED + base_idx * self.repeats + aug_idx
        random.seed(seed)
        np.random.seed(seed)

        tile = sample['tile']
        if isinstance(tile, torch.Tensor):
            arr = tile.detach().cpu().numpy()
        else:
            arr = tile.copy()

        arr_uint8 = (arr * 255.0).round().astype(np.uint8)

        if self.transform is not None:
            rec = self.transform(image=arr_uint8)
            aug_arr = rec['image'].astype(np.float32) / 255.0
        else:
            aug_arr = arr  # no transform

        sample['tile'] = aug_arr
        sample['subtiles'] = np.stack(
            split_into_subtiles(aug_arr, grid_size=self.grid_size),
            axis=0
        )
        return sample


# --------------------------------------------------------------------------------
# 4) Dataset holding pre-augmented list (same as before)
class StaticDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

# --------------------------------------------------------------------------------
# 5) Build a one-shot static dataset: original + repeats augmentations
#    Returns StaticDataset with same keys (including 'label', 'slide_idx', 'source_idx').
def build_static_dataset(base_ds: Dataset, repeats: int) -> StaticDataset:
    augmenter = AugmentFn(repeats)
    static_samples = []
    for base_idx in range(len(base_ds)):
        orig = deepcopy(base_ds[base_idx])  # includes all keys
        # append original
        static_samples.append(orig)
        # append repeats augmented versions
        for aug_idx in range(repeats):
            samp = deepcopy(orig)
            static_samples.append(augmenter(samp, base_idx, aug_idx))
    return StaticDataset(static_samples)

# --------------------------------------------------------------------------------
# 6) Augment grouped_data dict in-place style
#    Input: grouped_data: dict of lists (keys include 'tile','subtiles','label','slide_idx','source_idx',...)
#           image_keys: list of keys to augment (e.g. ['tile','subtiles'])
#           repeats: number of augmentations per sample
#    Returns new dict with same keys, length = original_N * (1 + repeats)
def augment_grouped_data(grouped_data: dict, image_keys: list[str], repeats: int = 1, mode="train") -> dict:
    """
    mode: "train", "val", "test"
    """
    keys = list(grouped_data.keys())
    N = len(grouped_data[keys[0]])
    new_data = {k: [] for k in keys}

    # 選擇對應transform
    if mode == "train" or mode == "val":
        transform = train_val_transform
    elif mode == "test":
        transform = test_transform
    else:
        raise ValueError(f"Unknown mode: {mode}")

    augmenter = AugmentFn(repeats=repeats, transform=transform)

    from copy import deepcopy as _deepcopy
    for i in range(N):
        sample = {k: grouped_data[k][i] for k in keys}
        # original (不做augmentation)
        for k in keys:
            new_data[k].append(sample[k])
        # repeats augmentations
        for aug_idx in range(repeats):
            samp = _deepcopy(sample)
            aug_samp = augmenter(samp, base_idx=i, aug_idx=aug_idx)
            for k in keys:
                if k in image_keys:
                    new_data[k].append(aug_samp[k])
                else:
                    new_data[k].append(sample[k])

    return new_data



def subset_grouped_data(data_dict, indices):
    return {k: [data_dict[k][i] for i in indices] for k in data_dict}


def plot_augmented_by_source(sample_ids, augmented_data, grouped_data=None):
    """
    Plot original and augmented tiles, subtiles, and labels for each source_idx in sample_ids.

    Args:
        sample_ids (list): list of source_idx values to visualize.
        augmented_data (dict): dict containing keys 'tile','subtiles','label','slide_idx','source_idx'.
        grouped_data (dict, optional): original data dict, not required for plotting.
    """
    for src in sample_ids:
        # Find all indices for this source (original + augmented)
        indices = [i for i, s in enumerate(augmented_data['source_idx']) if s == src]
        if not indices:
            continue
        n = len(indices)

        # Create figure with n rows, 3 columns
        fig = plt.figure(figsize=(15, 5 * n))
        gs = gridspec.GridSpec(n, 3,
                               width_ratios=[1, 1, 0.7],
                               hspace=0.4, wspace=0.3)

        for row, idx in enumerate(indices):
            tile = augmented_data['tile'][idx]        # (H,W,3)
            subs = augmented_data['subtiles'][idx]    # (9,Hs,Ws,3)
            label = np.array(augmented_data['label'][idx])
            slide_id = augmented_data['slide_idx'][idx]

            # 1) Tile
            ax0 = fig.add_subplot(gs[row, 0])
            ax0.imshow(tile)
            title = 'Original Tile' if row == 0 else f'Augmented {row}'
            ax0.set_title(title)
            ax0.axis('off')

            # 2) Subtiles mosaic
            ax1 = fig.add_subplot(gs[row, 1])
            Hs, Ws = subs.shape[1], subs.shape[2]
            big = np.zeros((3*Hs, 3*Ws, 3), dtype=subs.dtype)
            for i, patch in enumerate(subs):
                r, c = divmod(i, 3)
                big[r*Hs:(r+1)*Hs, c*Ws:(c+1)*Ws] = patch
            ax1.imshow(big)
            ax1.set_title('Subtiles 1–9', fontsize=12)
            ax1.axis('off')
            for i in range(9):
                r, c = divmod(i, 3)
                ax1.text(
                    c*Ws + 2, r*Hs + 2, str(i+1),
                    color='blue', fontsize=14, fontweight='bold',
                    backgroundcolor='white', alpha=0.7
                )

            # 3) Label bar
            ax2 = fig.add_subplot(gs[row, 2])
            ax2.bar(np.arange(label.shape[0]), label, color='tab:blue')
            ax2.set_title(f'Label (35-dim)\nslide_idx={slide_id}\nsource_idx={src}', fontsize=12)
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Value')
            ax2.set_xlim(-0.5, label.shape[0] - 0.5)

        fig.suptitle(f'Source_idx = {src}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()