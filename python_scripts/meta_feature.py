from torch.utils.data import DataLoader
from python_scripts.dataset_features import  *
from python_scripts.prediction_features import  *
import numpy as np

# === Main Function with Names ===
def generate_meta_features(dataset, model_for_recon, device, ae_type, oof_preds = None):
    """
    Generate meta-features and corresponding names.

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_features)
    names    : list of str, length n_features
    """

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 1) 收集所有 (feats, names) 到同一个 outputs 列表
    outputs = []

    # AE reconstruction loss
    feats, names = compute_ae_reconstruction_loss(model_for_recon, loader, device, ae_type)
    feats = feats[:, None]
    outputs.append((feats, names))

    # AE embeddings
    feats, names = compute_ae_embeddings(loader, model_for_recon, device)
    outputs.append((feats, names))

    # Latent stats
    latent_feats = outputs[1][0]
    feats, names = compute_latent_stats(latent_feats)
    outputs.append((feats, names))

    # RGB stats

    feats, names = compute_all_subtiles_rgb_stats(dataset)
    outputs.append((feats, names))
    feats, names = compute_subtiles_except_center_rgb_stats(dataset)
    outputs.append((feats, names))
    feats, names = compute_tile_rgb_stats(dataset)
    outputs.append((feats, names))
    feats, names = compute_subtile_contrast_stats(dataset)
    outputs.append((feats, names))

    # Texture & pattern features
    feats, names = compute_wavelet_stats(dataset)
    outputs.append((feats, names))
    feats, names = compute_sobel_stats(dataset)
    outputs.append((feats, names))

    # Color & distribution features
    feats, names = compute_hsv_stats(dataset)
    outputs.append((feats, names))
    feats, names = compute_color_moments(dataset)
    outputs.append((feats, names))

    # H&E stain features
    feats, names = compute_he_stats(dataset)
    outputs.append((feats, names))

    # Sliding-window std stats
    feats, names = compute_sliding_window_stats(dataset)
    outputs.append((feats, names))

    if oof_preds is not None:
        outputs.append((oof_preds, ['oof_preds']))
        feats, names = compute_entropy(oof_preds)
        outputs.append((feats, names))
        feats, names = compute_top2_diff(oof_preds)
        outputs.append((feats, names))        
        feats, names = compute_pairwise_diff(oof_preds)
        outputs.append((feats, names))
        feats, names = compute_dispersion(oof_preds)
        outputs.append((feats, names))
        
    # 2) unzip 成 feat_list 和 name_seq
    feat_list, name_seq = zip(*outputs)

    # 3) 逐块校验 feats 列数与 names 长度
    for feats, names_block in zip(feat_list, name_seq):
        ncols = feats.shape[1] if feats.ndim == 2 else 1
        if ncols != len(names_block):
            raise ValueError(
                f"Mismatch: got {ncols} columns but {len(names_block)} names "
                f"in block '{names_block[0].split('_')[0]}'"
            )
        print(
            f"{names_block[0].split('_')[0]:12s} -> cols: {ncols:4d}, names: {len(names_block):4d} OK"
        )

    # 4) 扁平化 names 并拼接 features
    name_list = [nm for block in name_seq for nm in block]
    features = np.concatenate(feat_list, axis=1)
    print(f"✅ Generated meta-features with shape: {features.shape}")

    return features, name_list