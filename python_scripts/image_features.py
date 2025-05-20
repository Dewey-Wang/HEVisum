import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.color import separate_stains, hed_from_rgb, rgb2hsv

import pywt

import joblib

from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis

from tqdm import tqdm




# === AE Reconstruction Loss ===
def compute_ae_reconstruction_loss(ae_model, dataloader, device, ae_type):
    """
    Returns
    -------
    losses : np.ndarray, shape (n_samples,)
    names  : list of str, e.g. ['ae_recon_loss_center']
    """
    ae_model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing AE recon loss"):
            tile = batch['tile'].to(device)
            subtiles = batch['subtiles'].to(device)
            recon = ae_model(tile, subtiles)

            if ae_type == 'center':
                target = subtiles[:, 4]
            else:
                target = subtiles

            loss = F.mse_loss(recon, target, reduction='none')
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
            losses.append(loss.cpu().numpy())
    arr = np.concatenate(losses)
    names = [f"ae-recon-loss_{ae_type}"]
    return arr, names

# === Latent Statistics ===
def compute_latent_stats(latents):
    """
    Returns
    -------
    feats : np.ndarray, shape (n_samples, 4)
    names : ['latent_mean', 'latent_std', 'latent_min', 'latent_max']
    """
    feats = np.concatenate([
        latents.mean(axis=1, keepdims=True),
        latents.std(axis=1, keepdims=True),
        latents.min(axis=1, keepdims=True),
        latents.max(axis=1, keepdims=True),
    ], axis=1)
    names = ['latent_mean', 'latent_std', 'latent_min', 'latent_max']
    return feats, names



# === All Subtiles RGB Stats ===
def compute_all_subtiles_rgb_stats(dataset):
    "Returns feats,names for all 9 subtiles"
    stats = []
    for item in dataset:
        subs = item['subtiles'].numpy()
        sample_stats = []
        for idx in range(subs.shape[0]):
            for ch in range(subs.shape[1]):
                vals = subs[idx, ch]
                sample_stats += [vals.mean(), vals.std(), vals.min(), vals.max()]
        stats.append(sample_stats)
    feats = np.array(stats)
    C = dataset[0]['subtiles'].shape[1]
    names = []
    for idx in range(9):
        for ch in range(C):
            for stat in ('mean','std','min','max'):
                names.append(f"subtile_{idx}_ch{ch}_{stat}")
    return feats, names

def compute_center_subtile_rgb_stats(dataset):
    """
    Returns feats,names for the center subtile (index 4) only.

    Each sample: for each channel in subtiles[4], compute mean, std, min, max.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, C*4)
    names : list of str, length C*4
    """
    stats = []
    for item in dataset:
        sub = item['subtiles'][4].numpy()  # (C, H, W)
        sample_stats = []
        for ch in range(sub.shape[0]):
            vals = sub[ch]
            sample_stats += [vals.mean(), vals.std(), vals.min(), vals.max()]
        stats.append(sample_stats)
    feats = np.array(stats)

    # build names
    C = dataset[0]['subtiles'][4].shape[0]
    names = []
    for ch in range(C):
        for stat in ('mean', 'std', 'min', 'max'):
            names.append(f"subtile4_ch{ch}_{stat}")

    return feats, names

# === Subtiles Except Center RGB Stats ===
def compute_subtiles_except_center_rgb_stats(dataset):
    per_subtile=False
    "Returns feats,names excluding center subtile"
    stats = []
    for item in dataset:
        subs = item['subtiles'].numpy()
        exclude = np.concatenate([subs[:4], subs[5:]], axis=0)
        sample_stats = []
        if per_subtile:
            for idx in range(exclude.shape[0]):
                for ch in range(exclude.shape[1]):
                    vals = exclude[idx, ch].flatten()
                    sample_stats += [vals.mean(), vals.std(), vals.min(), vals.max()]
        else:
            for ch in range(exclude.shape[1]):
                vals = exclude[:, ch].flatten()
                sample_stats += [vals.mean(), vals.std(), vals.min(), vals.max()]
        stats.append(sample_stats)
    feats = np.array(stats)
    C = dataset[0]['subtiles'].shape[1]
    names = []
    if per_subtile:
        for idx in range(8):
            for ch in range(C):
                for stat in ('mean','std','min','max'):
                    names.append(f"exsubtile-per-subtile_{idx}_ch{ch}_{stat}")
    else:
        for ch in range(C):
            for stat in ('mean','std','min','max'):
                names.append(f"exsubtiles_ch{ch}_{stat}")
    return feats, names


# === Tile RGB Stats ===
def compute_tile_rgb_stats(dataset):
    "Returns feats,names for full tile RGB stats"
    stats = []
    for item in dataset:
        tile = item['tile'].numpy()
        sample_stats = []
        for ch in range(tile.shape[0]):
            vals = tile[ch]
            sample_stats += [vals.mean(), vals.std(), vals.min(), vals.max()]
        stats.append(sample_stats)
    feats = np.array(stats)
    C = dataset[0]['tile'].shape[0]
    names = []
    for ch in range(C):
        for stat in ('mean','std','min','max'):
            names.append(f"tile_ch{ch}_{stat}")
    return feats, names

# === Wavelet Stats ===
def compute_wavelet_stats(dataset, wavelet='db1', level=2):
    """
    Compute wavelet stats (mean, std, min, max) for each sample's tile and all 9 subtiles.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_patches * n_coeffs * 4)
    names : list of str, length n_patches * n_coeffs * 4
    """
    # 1) Build name template using first sample
    patch_types = ['tile'] + [f'subtile_{j}' for j in range(9)]
    names = []
    sample0 = dataset[0]
    for p in patch_types:
        # choose array: tile or subtile j, channel 0
        arr0 = sample0['tile'].numpy()[0] if p == 'tile' else sample0['subtiles'][int(p[-1])].numpy()[0]
        coeffs0 = pywt.wavedec2(arr0, wavelet=wavelet, level=level)
        for lvl, coeff in enumerate(coeffs0):
            if isinstance(coeff, tuple):
                for bidx, sub in enumerate(coeff):
                    for stat in ('mean','std','min','max'):
                        names.append(f"wavelet_{p}_level{lvl}_band{bidx}_{stat}")
            else:
                for stat in ('mean','std','min','max'):
                    names.append(f"wavelet-{p}_approx_{stat}")

    # 2) Compute features for each sample
    feats_list = []
    for item in dataset:
        vals = []
        # full tile
        arr = item['tile'].numpy()[0]
        coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                for sub in coeff:
                    vals += [sub.mean(), sub.std(), sub.min(), sub.max()]
            else:
                vals += [coeff.mean(), coeff.std(), coeff.min(), coeff.max()]
        # each subtile
        subs = item['subtiles'].numpy()
        for j in range(subs.shape[0]):
            arr = subs[j][0]
            coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
            for coeff in coeffs:
                if isinstance(coeff, tuple):
                    for sub in coeff:
                        vals += [sub.mean(), sub.std(), sub.min(), sub.max()]
                else:
                    vals += [coeff.mean(), coeff.std(), coeff.min(), coeff.max()]
        feats_list.append(vals)

    feats = np.array(feats_list)
    return feats, names

# def compute_wavelet_stats(dataset, wavelet='db1', level=2):
#     """
#     Returns
#     -------
#     feats : np.ndarray, shape (n_samples, n_coeffs*2)
#         每个样本的小波系数均值和标准差。
#     names : list of str, length n_coeffs*2
#         ['wavelet_approx_mean','wavelet_approx_std',
#          'wavelet_level1_band0_mean','wavelet_level1_band0_std', … ]
#     """
#     import pywt
#     # —— 1. 用第一个样本生成 names 模板 —— 
#     first_patch = dataset[0]['subtiles'][4].numpy()[0]
#     coeffs_template = pywt.wavedec2(first_patch, wavelet=wavelet, level=level)
#     names = []
#     for lvl, arr in enumerate(coeffs_template):
#         if isinstance(arr, tuple):
#             for bidx, sub in enumerate(arr):
#                 names += [
#                     f"wavelet_level{lvl}_band{bidx}_mean",
#                     f"wavelet_level{lvl}_band{bidx}_std",
#                 ]
#         else:
#             names += [
#                 "wavelet_approx_mean",
#                 "wavelet_approx_std",
#             ]

#     # —— 2. 遍历所有样本只拼数值 —— 
#     feats_list = []
#     for item in dataset:
#         patch = item['subtiles'][4].numpy()[0]
#         coeffs = pywt.wavedec2(patch, wavelet=wavelet, level=level)
#         vals = []
#         for arr in coeffs:
#             if isinstance(arr, tuple):
#                 for sub in arr:
#                     vals.append(sub.mean())
#                     vals.append(sub.std())
#             else:
#                 vals.append(arr.mean())
#                 vals.append(arr.std())
#         feats_list.append(vals)

#     feats = np.array(feats_list)  # shape (n_samples, len(names))
#     return feats, names


def compute_sobel_stats(dataset):
    """
    Compute Sobel edge statistics (mean, std, min, max) for each sample's tile and all 9 subtiles.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_patches * 4)
    names : list of str, length n_patches * 4
    """
    # Define patch types: full tile and 9 subtiles
    patch_types = ['tile'] + [f'subtile_{j}' for j in range(9)]
    # Build names template
    names = []
    for p in patch_types:
        for stat in ('mean', 'std', 'min', 'max'):
            names.append(f"sobel-{p}_{stat}")

    # Compute features
    feats_list = []
    for item in dataset:
        vals = []
        # Full tile
        img_tile = item['tile'].numpy()  # shape (C, H, W)
        # convert to grayscale by mean across channels
        gray = img_tile.mean(axis=0)
        edge = sobel(gray)
        vals += [edge.mean(), edge.std(), edge.min(), edge.max()]
        # Each subtile
        subtiles = item['subtiles'].numpy()  # shape (9, C, H, W)
        for j in range(subtiles.shape[0]):
            img_sub = subtiles[j]
            gray = img_sub.mean(axis=0)
            edge = sobel(gray)
            vals += [edge.mean(), edge.std(), edge.min(), edge.max()]
        feats_list.append(vals)

    feats = np.array(feats_list)
    return feats, names
# @register_feature
# def compute_sobel_stats(dataset):
#     "Returns feats,names of sobel edge stats"
#     feats, names = [], ['sobel_mean','sobel_std','sobel_min','sobel_max']
#     for item in dataset:
#         gray = item['tile'].numpy().mean(axis=0)
#         edge = sobel(gray)
#         feats.append([edge.mean(), edge.std(), edge.min(), edge.max()])
#     return np.array(feats), names

# === HSV Stats ===
def compute_hsv_stats(dataset):
    """
    Compute HSV channel stats (mean, std, min, max) for each sample's tile and all 9 subtiles.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_patches * 4)
    names : list of str, length n_patches * 4
    """
    patch_types = ['tile'] + [f'subtile_{j}' for j in range(9)]
    # build names
    names = []
    for p in patch_types:
        for c in ('H','S','V'):
            for stat in ('mean','std','min','max'):
                names.append(f"hsv-{p}_{c}_{stat}")
    # compute feats
    feats_list = []
    for item in dataset:
        vals = []
        # tile
        img = item['tile'].numpy()
        gray_img = img[:3].transpose(1,2,0)
        hsv_img = rgb2hsv(gray_img)
        for c in range(3):
            comp = hsv_img[:,:,c]
            vals += [comp.mean(), comp.std(), comp.min(), comp.max()]
        # subtiles
        subs = item['subtiles'].numpy()
        for j in range(subs.shape[0]):
            gray_sub = subs[j][:3].transpose(1,2,0)
            hsv_sub = rgb2hsv(gray_sub)
            for c in range(3):
                comp = hsv_sub[:,:,c]
                vals += [comp.mean(), comp.std(), comp.min(), comp.max()]
        feats_list.append(vals)
    feats = np.array(feats_list)
    return feats, names

# def compute_hsv_stats(dataset):
#     "Returns feats,names of HSV channel stats"
#     feats_list = []
#     for item in dataset:
#         sub = item['subtiles'][4].numpy()
#         img = sub[:3].transpose(1,2,0)
#         hsv = rgb2hsv(img)
#         sample = []
#         for ch, cname in enumerate(('H','S','V')):
#             vals = hsv[:,:,ch]
#             sample += [vals.mean(), vals.std(), vals.min(), vals.max()]
#         feats_list.append(sample)
#     feats = np.array(feats_list)
#     names = []
#     for cname in ('H','S','V'):
#         for stat in ('mean','std','min','max'):
#             names.append(f"hsv_{cname}_{stat}")
#     return feats, names


# === Color Moments ===
def compute_color_moments(dataset):
    """
    Compute color moments (mean, std, skew, kurtosis) for each sample's tile and all 9 subtiles.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_patches * 4)
    names : list of str, length n_patches * 4
    """
    patch_types = ['tile'] + [f'subtile_{j}' for j in range(9)]
    # build names
    names = []
    for p in patch_types:
        for ch in range(dataset[0]['tile'].numpy().shape[0]):
            for stat in ('mean','std','skew','kurtosis'):
                names.append(f"color-{p}_ch{ch}_{stat}")
    # compute feats
    feats_list = []
    for item in dataset:
        vals = []
        # tile
        img = item['tile'].numpy().transpose(1,2,0)  # H, W, C
        for ch in range(img.shape[2]):
            arr = img[:,:,ch].ravel()
            vals += [arr.mean(), arr.std(), skew(arr), kurtosis(arr)]
        # subtiles
        subs = item['subtiles'].numpy()
        for j in range(subs.shape[0]):
            sub_img = subs[j].transpose(1,2,0)
            for ch in range(sub_img.shape[2]):
                arr = sub_img[:,:,ch].ravel()
                vals += [arr.mean(), arr.std(), skew(arr), kurtosis(arr)]
        feats_list.append(vals)
    feats = np.array(feats_list)
    return feats, names

# def compute_color_moments(dataset):
#     "Returns feats,names of color moments stats"
#     feats_list = []
#     for item in dataset:
#         sub = item['subtiles'][4].numpy().transpose(1,2,0)
#         sample = []
#         for ch in range(sub.shape[2]):
#             vals = sub[:,:,ch].ravel()
#             sample += [vals.mean(), vals.std(), skew(vals), kurtosis(vals)]
#         feats_list.append(sample)
#     feats = np.array(feats_list)
#     C = dataset[0]['subtiles'][4].numpy().shape[0]
#     names = []
#     for ch in range(C):
#         for stat in ('mean','std','skew','kurtosis'):
#             names.append(f"color_{ch}_{stat}")
#     return feats, names


# === Sliding Window Stats ===
def compute_sliding_window_stats(dataset, window_size=4, stride=2):
    "Returns feats,names of sliding window contrast stats"
    feats_list = []
    for item in dataset:
        sub = item['subtiles'][4].numpy()
        C, H, W = sub.shape
        row = []
        for ch in range(C):
            for i in range(0, H-window_size+1, stride):
                for j in range(0, W-window_size+1, stride):
                    win = sub[ch, i:i+window_size, j:j+window_size]
                    #row += [win.mean(), win.std(), win.max(), win.min()]
                    row += [win.std()]

        feats_list.append(row)
    feats = np.array(feats_list)
    # build names
    C = dataset[0]['subtiles'][4].numpy().shape[0]
    nx = (H - window_size)//stride + 1
    ny = (W - window_size)//stride + 1
    names = []
    for ch in range(C):
        for i in range(nx):
            for j in range(ny):
                #for stat in ('mean','std','max','min'):
                    # names.append(f"locstd_{stat}_ch{ch}_i{i}_j{j}")
                names.append(f"locstd_std_ch{ch}_i{i}_j{j}")

    return feats, names

# === H&E Stats ===
def compute_he_stats(dataset):
    """
    Compute H&E stain stats (mean, std, min, max) for each sample's tile and all 9 subtiles.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_patches * 8)
    names : list of str, length n_patches * 8
    """
    patch_types = ['tile'] + [f'subtile_{j}' for j in range(9)]
    # build names
    names = []
    for p in patch_types:
        for stain in ('H', 'E'):
            for stat in ('mean', 'std', 'min', 'max'):
                names.append(f"he-{p}_{stain}_{stat}")

    feats_list = []
    for item in dataset:
        vals = []
        # full tile
        img_tile = item['tile'].numpy()[:3].transpose(1,2,0)
        hed_tile = separate_stains(img_tile, hed_from_rgb)
        for idx, stain in enumerate(('H','E')):
            comp = hed_tile[:,:,idx]
            vals += [comp.mean(), comp.std(), comp.min(), comp.max()]
        # each subtile
        subs = item['subtiles'].numpy()
        for j in range(subs.shape[0]):
            img_sub = subs[j][:3].transpose(1,2,0)
            hed_sub = separate_stains(img_sub, hed_from_rgb)
            for idx in range(2):
                comp = hed_sub[:,:,idx]
                vals += [comp.mean(), comp.std(), comp.min(), comp.max()]
        feats_list.append(vals)

    feats = np.array(feats_list)
    return feats, names


# def compute_he_stats(dataset):
#     "Returns feats,names of H&E stain stats"
#     feats, names = [], ['he_H_mean','he_H_std','he_H_min','he_H_max',
#                          'he_E_mean','he_E_std','he_E_min','he_E_max']
#     for item in dataset:
#         sub = item['subtiles'][4].numpy()[:3].transpose(1,2,0)
#         hed = separate_stains(sub, hed_from_rgb)
#         h, e = hed[:,:,0], hed[:,:,1]
#         feats.append([h.mean(), h.std(), h.min(), h.max(),
#                       e.mean(), e.std(), e.min(), e.max()])
#     return np.array(feats), names




# === AE Embeddings ===
def compute_ae_embeddings(loader, recon_model, device):
    """
    Returns
    -------
    embeddings : np.ndarray, shape (n_samples, fusion_dim)
    names      : ["ae_emb0", ..., f"ae_emb{fusion_dim-1}"]
    """
    recon_model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            tiles = batch['tile'].to(device)
            subtiles = batch['subtiles'].to(device)
            subtiles = subtiles.contiguous()
            tiles     = tiles.contiguous()
            f_c = recon_model.enc_center(subtiles[:, 4])
            f_n = recon_model.enc_neigh(subtiles)
            f_t = recon_model.enc_tile(tiles)
            fused = torch.cat([f_c, f_n, f_t], dim=1)
            embs.append(fused.cpu().numpy())
    embeddings = np.vstack(embs)
    fusion_dim = embeddings.shape[1]
    names = [f"ae_emb{i}" for i in range(fusion_dim)]
    return embeddings, names


def compute_subtile_contrast_stats(dataset):
    """
    Compute contrast between center subtile and surrounding subtiles:
    For each sample and each channel, calculate:
      diff = center_mean - surround_mean
      ratio = center_mean / (surround_mean + eps)

    Returns
    -------
    feats : np.ndarray, shape (n_samples, C*2)
        Columns are [diff_ch0, ..., diff_ch{C-1}, ratio_ch0, ..., ratio_ch{C-1}]
    names : list of str, length C*2
        ['contrast_diff_ch0', ..., 'contrast_diff_ch{C-1}',
         'contrast_ratio_ch0', ..., 'contrast_ratio_ch{C-1}']
    """
    stats = []
    for item in dataset:
        subs = item['subtiles'].numpy()           # (9, C, H, W)
        center = subs[4]                          # (C, H, W)
        surround = np.concatenate([subs[:4], subs[5:]], axis=0)  # (8, C, H, W)

        # per-channel means
        C = center.shape[0]
        center_mean = center.reshape(C, -1).mean(axis=1)
        # first average spatial dims, then average over the 8 surroundings
        surround_mean = surround.reshape(8, C, -1).mean(axis=2).mean(axis=0)

        diff  = center_mean - surround_mean

        stats.append(diff)

    feats = np.array(stats)  # shape (n_samples, C*2)

    # build names
    names = []
    for ch in range(C):
        names.append(f"contrast_diff_ch{ch}")


    return feats, names


__all__ = [
    "compute_ae_reconstruction_loss",
    "compute_latent_stats",
    "compute_all_subtiles_rgb_stats",
    "compute_subtiles_except_center_rgb_stats",
    "compute_tile_rgb_stats",
    "compute_subtile_contrast_stats",
    "compute_wavelet_stats",
    "compute_sobel_stats",
    "compute_hsv_stats",
    "compute_color_moments",
    "compute_he_stats",
    "compute_ae_embeddings",
    "compute_sliding_window_stats",
    "compute_center_subtile_rgb_stats"
]