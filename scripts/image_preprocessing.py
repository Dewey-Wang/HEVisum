import numpy as np
from torchvision import transforms
import torch
import torchstain
from typing import Callable

def make_tissue_mask(img: np.ndarray, thresh: float = 0.8) -> np.ndarray:
    gray = img.mean(axis=2)
    return gray < thresh

def sample_tissue_patches_np(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    n_patches: int = 5,
    min_frac: float = 0.5,
    seed: int = 0
) -> list[np.ndarray]:
    """
    Randomly sample n_patches of size patch_size from img,
    requiring that at least min_frac of each patch is tissue.
    Uses seed for reproducibility.
    """
    H, W, _ = img.shape
    rng = np.random.RandomState(seed)
    out = []
    tries = 0
    while len(out) < n_patches and tries < n_patches * 50:
        i = rng.randint(0, H - patch_size)
        j = rng.randint(0, W - patch_size)
        subm = mask[i:i+patch_size, j:j+patch_size]
        if subm.mean() >= min_frac:
            out.append(img[i:i+patch_size, j:j+patch_size])
        tries += 1
    return out


def normalize_np_image(
    img_np: np.ndarray,
    to_tensor_fn: Callable[[np.ndarray], torch.Tensor],
    normalizer: torchstain.normalizers.MacenkoNormalizer,
) -> np.ndarray:
    """
    img_np: H×W×3 float32 in [0,1]
    to_tensor_fn: maps uint8 H×W×3 → C×H×W Tensor in [0,255]
    normalizer: a fitted MacenkoNormalizer
    Returns: H×W×3 uint8 in [0,255]
    """
    # to uint8
    img_u8 = (img_np * 255).astype(np.uint8)

    # to tensor [0,255]
    t = to_tensor_fn(img_u8)                        # C×H×W

    # macenko normalize → still [0,255]
    norm_t, _, _ = normalizer.normalize(I=t, stains=True)

    # back to H×W×C uint8
    norm_np = (
        norm_t
        .clamp(0, 255)
        .permute(0, 1, 2)  # C,H,W → H,W,C
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return norm_np
