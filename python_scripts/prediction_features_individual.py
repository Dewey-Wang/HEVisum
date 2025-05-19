import numpy as np
from itertools import combinations

def compute_self_vs_others_diff(oof_preds: np.ndarray, cell_idx: int):
    """
    对每个样本，计算指定 cell_idx 的预测值
    与其它所有 j≠cell_idx 的预测值差值 (p_i - p_j)。

    Returns
    -------
    feats : (n_samples, C-1)
    names : ['self_diff_to_0', ...]
    """
    n, C = oof_preds.shape
    pi = oof_preds[:, cell_idx:cell_idx+1]  # (n,1)
    diffs = []
    names = []
    for j in range(C):
        if j == cell_idx: continue
        diffs.append(pi - oof_preds[:, j:j+1])
        names.append(f"diff-self_{cell_idx}_minus_{j}")
    feats = np.hstack(diffs)
    return feats, names


def compute_self_vs_others_ratio(oof_preds: np.ndarray, cell_idx: int, eps=1e-12):
    """
    对每个样本，计算指定 cell_idx 的预测值
    与其它所有 j≠cell_idx 的预测值比值 (p_i / (p_j + eps))。

    Returns
    -------
    feats : (n_samples, C-1)
    names : ['self_ratio_to_0', ...]
    """
    n, C = oof_preds.shape
    pi = oof_preds[:, cell_idx:cell_idx+1]
    ratios = []
    names = []
    for j in range(C):
        if j == cell_idx: continue
        ratios.append(pi / (oof_preds[:, j:j+1] + eps))
        names.append(f"ratio-self_{cell_idx}_ratio_{j}")
    feats = np.hstack(ratios)
    return feats, names

def compute_normalized_rank(oof_preds: np.ndarray, cell_idx: int):
    """
    对每个样本，对预测值降序排位，然后把第 cell_idx 类的位次归一化到 [0,1]。
    """
    # argsort 默认升序，前面加 - 变降序
    ranks = np.argsort(-oof_preds, axis=1)
    # 把 ranks[i,j] 从 0..C-1 转成 pos[j]
    # 也可以用: inv_rank[i, ranks[i,k]] = k
    n, C = oof_preds.shape
    inv = np.zeros_like(ranks)
    for i in range(n):
        inv[i, ranks[i]] = np.arange(C)
    pos = inv[:, cell_idx]
    norm = pos / (C - 1)
    return norm.reshape(-1,1), [f"rank-self_{cell_idx}_norm_rank"]

def compute_self_zscore(oof_preds: np.ndarray, cell_idx: int, eps: float = 1e-12):
    """
    对每个样本，计算 p_i 的 z-score = (p_i - mean(p_all)) / std(p_all)
    """
    row_mean = oof_preds.mean(axis=1, keepdims=True)
    row_std  = oof_preds.std(axis=1, keepdims=True) + eps
    z = (oof_preds[:, cell_idx:cell_idx+1] - row_mean) / row_std
    return z, [f"zscore-self_{cell_idx}_zscore"]

def compute_self_vs_rank_diff(oof_preds: np.ndarray, cell_idx: int):
    """
    对每个样本，先对预测值降序排序，
    计算指定 cell 在排序中的位置 k，
    然后输出它与前后两位的差值 (p_k - p_{k±1})。
    如果它在首尾，只输出一侧差值。

    Returns
    -------
    feats : (n_samples, up to 2)
    names : ['self_rank_diff_prev','self_rank_diff_next']
    """
    n, C = oof_preds.shape
    sorted_idx = np.argsort(-oof_preds, axis=1)  # (n,C)
    diffs = []
    names = []
    for i in range(n):
        order = sorted_idx[i]
        pos = np.where(order == cell_idx)[0][0]
        row = oof_preds[i]
        if pos > 0:
            diffs.append(row[cell_idx] - row[order[pos-1]])
            names.append("self-rank-diff-prev")
        if pos < C-1:
            diffs.append(row[cell_idx] - row[order[pos+1]])
            names.append("self-rank-diff-next")
    # 由于前后可能各有，有时一侧不存在，所以均匀填充 NaN
    out = np.full((n, len(names)), np.nan)
    for i in range(n):
        order = sorted_idx[i]
        pos = np.where(order == cell_idx)[0][0]
        col = 0
        if pos>0:
            out[i,col] = oof_preds[i,cell_idx] - oof_preds[i, order[pos-1]]
            col+=1
        if pos<C-1:
            out[i,col] = oof_preds[i,cell_idx] - oof_preds[i, order[pos+1]]
    return out, names


def compute_self_topk_stats(oof_preds: np.ndarray, cell_idx: int, k: int):
    """
    对每个样本，先对预测值降序排序，
    找出指定 cell_idx 在排序中的位置 pos，
    然后取 pos 前 k 个和后 k 个（若不足则截断），
    分别计算这些邻近值的均值/方差。

    Returns
    -------
    feats : (n_samples, 4)   # 前 k mean/std + 后 k mean/std
    names : ['self_pre_k_mean','self_pre_k_std','self_next_k_mean','self_next_k_std']
    """
    n, C = oof_preds.shape
    sorted_idx = np.argsort(-oof_preds, axis=1)
    out = np.zeros((n,4))
    for i in range(n):
        order = sorted_idx[i]
        pos = np.where(order==cell_idx)[0][0]
        # 前 k
        pre = order[max(0,pos-k):pos]
        next_ = order[pos+1:pos+1+k]
        pre_vals = oof_preds[i, pre] if len(pre)>0 else np.array([np.nan])
        next_vals = oof_preds[i, next_] if len(next_)>0 else np.array([np.nan])
        out[i,0] = np.nanmean(pre_vals)
        out[i,1] = np.nanstd(pre_vals)
        out[i,2] = np.nanmean(next_vals)
        out[i,3] = np.nanstd(next_vals)
    names = [f"self_pre{k}_mean", f"self-pre{k}-std",
             f"self_next{k}_mean",f"self-next{k}-std"]
    return out, names

import numpy as np

def compute_self_adjacent_diffs(oof_preds: np.ndarray, cell_idx: int, eps: float = 1e-12):
    """
    对每个样本，找到自己 (cell_idx) 在该行降序分布里的位置，
    然后计算：
      - diff_up   = p_self - p[next higher rank]   (如果自己已是最高则设 0)
      - diff_down = p_self - p[next lower  rank]   (如果自己已是最低则设 0)

    Returns
    -------
    feats : np.ndarray, shape (n_samples, 2)
    names : ['self{cell_idx}_diff_up', 'self{cell_idx}_diff_down']
    """
    n, C = oof_preds.shape
    # 1) 排序并记录每行的降序索引 & 反向 rank
    sorted_idx = np.argsort(-oof_preds, axis=1)  # 每行降序后各元素的原始列下标
    # inv_rank[i,j] = 在第 i 行中，第 j 类是第几大的位置（0 最大）
    inv_rank = np.zeros_like(sorted_idx)
    for i in range(n):
        inv_rank[i, sorted_idx[i]] = np.arange(C)

    # 2) 对每行，针对 cell_idx，找它的 rank 位置
    self_rank = inv_rank[:, cell_idx]  # shape (n,)

    p_self = oof_preds[:, cell_idx]
    diff_up   = np.zeros(n, dtype=float)
    diff_down = np.zeros(n, dtype=float)

    # 3) 对每个样本：
    #    如果 self_rank > 0，则上一个更大的是 sorted_idx[i, self_rank-1]
    #    如果 self_rank < C-1，则下一个更小的是 sorted_idx[i, self_rank+1]
    for i in range(n):
        r = self_rank[i]
        if r > 0:
            idx_up = sorted_idx[i, r-1]
            diff_up[i] = p_self[i] - oof_preds[i, idx_up]
        if r < C-1:
            idx_dn = sorted_idx[i, r+1]
            diff_down[i] = p_self[i] - oof_preds[i, idx_dn]

    feats = np.stack([diff_up, diff_down], axis=1)
    names = [f"adj-self-up_{cell_idx}_diff", f"adj-self-donw_{cell_idx}_diff"]
    return feats, names

def compute_self_adjacent_ratio(oof_preds: np.ndarray, cell_idx: int, eps: float = 1e-12):
    """
    同上，但计算比率：
      - ratio_up   = p_self / (p[next higher] + eps)
      - ratio_down = p_self / (p[next lower]  + eps)
    """
    n, C = oof_preds.shape
    sorted_idx = np.argsort(-oof_preds, axis=1)
    inv_rank = np.zeros_like(sorted_idx)
    for i in range(n):
        inv_rank[i, sorted_idx[i]] = np.arange(C)

    self_rank = inv_rank[:, cell_idx]
    p_self = oof_preds[:, cell_idx]
    ratio_up   = np.ones(n, dtype=float)
    ratio_down = np.ones(n, dtype=float)

    for i in range(n):
        r = self_rank[i]
        if r > 0:
            up = oof_preds[i, sorted_idx[i, r-1]]
            ratio_up[i] = p_self[i] / (up + eps)
        if r < C-1:
            dn = oof_preds[i, sorted_idx[i, r+1]]
            ratio_down[i] = p_self[i] / (dn + eps)

    feats = np.stack([ratio_up, ratio_down], axis=1)
    names = [f"adj-ratio-self-up_{cell_idx}", f"adj-ratio-self-down_{cell_idx}"]
    return feats, names

__all__ = [
    # 新增的 self-centric 特征
    "compute_self_vs_others_diff",
    "compute_self_vs_others_ratio",
    "compute_normalized_rank",
    "compute_self_zscore",
    "compute_self_vs_rank_diff",
    "compute_self_topk_stats",
    "compute_self_adjacent_diffs",
    "compute_self_adjacent_ratio",
]