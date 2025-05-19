import numpy as np
from itertools import combinations
import numpy as np
from scipy.signal import find_peaks, peak_widths
from itertools import combinations
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
def compute_entropy(oof_preds):
    """
    Compute Shannon entropy for each sample's OOF prediction distribution.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, 1)
    names : ['entropy']
    """
    # normalize to sum 1
    probs = oof_preds / (oof_preds.sum(axis=1, keepdims=True))
    ent = -np.sum(probs * np.log(probs), axis=1, keepdims=True)
    return ent, ['entropy']

def compute_topn_stats_multi(oof_preds, max_n):
    """
    Compute summary stats (mean, std, min, max) for top-k values for k = max_n, max_n-1, ..., 2.

    Parameters
    ----------
    oof_preds : np.ndarray, shape (n_samples, C)
    max_n     : int, upper bound k (must satisfy 2 <= max_n <= C)

    Returns
    -------
    feats : np.ndarray, shape (n_samples, (max_n-1)*4)
    names : list of str, length (max_n-1)*4
    """
    n_samples, C = oof_preds.shape
    if not (2 <= max_n <= C):
        raise ValueError(f"max_n must be between 2 and {C}")
    sorted_preds = -np.sort(-oof_preds, axis=1)
    blocks = []
    names = []
    for k in range(max_n, 1, -1):  # k = max_n, max_n-1, ..., 2
        topk = sorted_preds[:, :k]
        mean_ = topk.mean(axis=1, keepdims=True)
        std_  = topk.std(axis=1, keepdims=True)
        min_  = topk.min(axis=1, keepdims=True)
        max_  = topk.max(axis=1, keepdims=True)
        blocks.append(np.hstack([mean_, std_, min_, max_]))
        for stat in ('mean','std','min','max'):
            names.append(f"top_{k}_{stat}")
    feats = np.hstack(blocks)
    return feats, names

def compute_lastn_stats_multi(oof_preds, max_n):
    """
    Compute summary stats (mean, std, min, max) for bottom-k values for k = max_n, max_n-1, ..., 2.

    Parameters
    ----------
    oof_preds : np.ndarray, shape (n_samples, C)
    max_n     : int, upper bound k (must satisfy 2 <= max_n <= C)

    Returns
    -------
    feats : np.ndarray, shape (n_samples, (max_n-1)*4)
    names : list of str, length (max_n-1)*4
    """
    n_samples, C = oof_preds.shape
    if not (2 <= max_n <= C):
        raise ValueError(f"max_n must be between 2 and {C}")
    sorted_preds = np.sort(oof_preds, axis=1)  # ascending
    blocks = []
    names = []
    for k in range(max_n, 1, -1):
        lastk = sorted_preds[:, -k:]
        mean_ = lastk.mean(axis=1, keepdims=True)
        std_  = lastk.std(axis=1, keepdims=True)
        min_  = lastk.min(axis=1, keepdims=True)
        max_  = lastk.max(axis=1, keepdims=True)
        blocks.append(np.hstack([mean_, std_, min_, max_]))
        for stat in ('mean','std','min','max'):
            names.append(f"last_{k}_{stat}")
    feats = np.hstack(blocks)
    return feats, names

def compute_adjacent_diffs(oof_preds, stride=1):
    """
    Compute differences between sorted predictions at positions i and i+stride.

    For each sample, sorts predictions descending and then for each i computes:
        diff_i = sorted_preds[:, i] - sorted_preds[:, i+stride]

    Parameters
    ----------
    oof_preds : np.ndarray, shape (n_samples, C)
    stride    : int, gap between indices (default 1 for adjacent)

    Returns
    -------
    feats : np.ndarray, shape (n_samples, C-stride)
    names : list of str, e.g. ['adj_diff_0_1', 'adj_diff_1_2', ...]
    """
    # sort descending
    sorted_preds = -np.sort(-oof_preds, axis=1)
    n_samples, C = sorted_preds.shape
    # for each i, compute difference with i+stride
    diffs = np.stack([
        sorted_preds[:, i] - sorted_preds[:, i + stride]
        for i in range(C - stride)
    ], axis=1)  # shape (n_samples, C-stride)
    names = [f"adj_diff_{i}_{i+stride}" for i in range(C - stride)]
    return diffs, names


from itertools import combinations

def compute_pairwise_diff(oof_preds):
    """
    Compute raw differences for every pair of cell-type predictions.
    For each sample, returns array of length C*(C-1)/2 in order of (i<j).

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_pairs)
    names : list of str, e.g. ['pw_0_1', 'pw_0_2', 'pw_1_2', ...]
    """
    n_samples, C = oof_preds.shape
    idx_pairs = list(combinations(range(C), 2))
    diffs = np.stack([oof_preds[:, i] - oof_preds[:, j] for i, j in idx_pairs], axis=1)
    names = [f"pw_{i}_{j}" for i, j in idx_pairs]
    return diffs, names


def compute_dispersion(oof_preds):
    """
    Compute dispersion metric (Gini impurity) across cell-type predictions.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, 1)
    names : ['dispersion']
    """
    probs = oof_preds / (oof_preds.sum(axis=1, keepdims=True))
    gini = 1 - np.sum(probs**2, axis=1, keepdims=True)
    return gini, ['dispersion']

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import combinations

def compute_median_mad(oof_preds):
    """
    Returns median and MAD (median absolute deviation) per sample.
    """
    med = np.median(oof_preds, axis=1, keepdims=True)
    mad = np.median(np.abs(oof_preds - med), axis=1, keepdims=True)
    names = ["mad"]
    return mad, names

def compute_skew_kurt(oof_preds):
    """
    Returns skewness and kurtosis per sample.
    """
    sk = skew(oof_preds, axis=1).reshape(-1,1)
    kt = kurtosis(oof_preds, axis=1).reshape(-1,1)
    feats = np.hstack([sk, kt])
    names = ["skewness", "kurtosis"]
    return feats, names

def compute_percentile_iqr(oof_preds, percentiles=[25,50,75]):
    """
    Returns specified percentiles and IQR (75th-25th) per sample.
    """
    ps = np.percentile(oof_preds, percentiles, axis=1).T  # shape (n_samples, len(percentiles))
    iqr = (ps[:,percentiles.index(75)] - ps[:,percentiles.index(25)]).reshape(-1,1)
    feats = np.hstack([ps, iqr])
    names = [f"p{p}" for p in percentiles] + ["iqr"]
    return feats, names

def compute_renyi_entropy(oof_preds, alpha=2, eps=1e-12):
    """
    Rényi entropy of order alpha > 0 (alpha != 1).
    """
    p = np.clip(oof_preds / (oof_preds.sum(axis=1,keepdims=True)+eps), eps, 1)
    if alpha == 1:
        return compute_entropy(oof_preds)
    H = (1/(1-alpha)) * np.log(np.sum(p**alpha, axis=1)+eps)
    return H.reshape(-1,1), [f"renyi_entropy_{alpha}"]

def compute_kl_uniform(oof_preds, eps=1e-12):
    """
    KL divergence from uniform distribution.
    """
    n_classes = oof_preds.shape[1]
    p = oof_preds / (oof_preds.sum(axis=1,keepdims=True)+eps)
    u = 1.0/n_classes
    kl = np.sum(p * np.log((p+eps)/u), axis=1, keepdims=True)
    return kl, ["kl_uniform"]

def compute_js_uniform(oof_preds, eps=1e-12):
    """
    Jensen-Shannon divergence to uniform.
    """
    n = oof_preds.shape[1]
    p = oof_preds / (oof_preds.sum(axis=1,keepdims=True)+eps)
    u = np.full_like(p, 1.0/n)
    m = 0.5*(p+u)
    js = 0.5*(np.sum(p*np.log((p+eps)/m), axis=1)+
              np.sum(u*np.log((u+eps)/m), axis=1))
    return js.reshape(-1,1), ["js_uniform"]

def compute_hellinger_uniform(oof_preds):
    """
    Hellinger distance to uniform.
    """
    n = oof_preds.shape[1]
    p = oof_preds / (oof_preds.sum(axis=1,keepdims=True)+1e-12)
    u = np.full_like(p, 1.0/n)
    dist = np.sqrt(0.5 * np.sum((np.sqrt(p)-np.sqrt(u))**2, axis=1))
    return dist.reshape(-1,1), ["hellinger_uniform"]


def compute_mass_topk(oof_preds, k):
    """
    Sum of top-k probabilities per sample.
    """
    topk = -np.sort(-oof_preds, axis=1)[:, :k]
    mass = np.sum(topk, axis=1, keepdims=True)
    return mass, [f"mass_top_{k}"]

def compute_tail_mass(oof_preds, k):
    """
    Sum of bottom-k probabilities per sample.
    """
    bottomk = np.sort(oof_preds, axis=1)[:, :k]
    mass = np.sum(bottomk, axis=1, keepdims=True)
    return mass, [f"mass_tail_{k}"]

def compute_cdf_slope(oof_preds):
    """
    Fit a line to the cumulative sum of sorted predictions;
    return the slope coefficient per sample.
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    cdf = np.cumsum(sorted_p, axis=1)
    x = np.arange(1, sorted_p.shape[1]+1)
    # slope = cov(x, cdf)/var(x)
    x_mean = x.mean()
    var_x = ((x - x_mean)**2).sum()
    slopes = []
    for row in cdf:
        cov = ((x - x_mean)*(row - row.mean())).sum()
        slopes.append(cov/var_x)
    return np.array(slopes).reshape(-1,1), ["cdf_slope"]

def compute_pca_components(oof_preds, n_components=10):
    """
    PCA on OOF_preds; return first n_components.
    """
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(oof_preds)
    names = [f"pca_{i}" for i in range(n_components)]
    return comps, names

def compute_kmeans_features(oof_preds, n_clusters=5):
    """
    KMeans cluster ID and distances to each center.
    """
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(oof_preds)
    labels = km.labels_.reshape(-1,1)
    dists = km.transform(oof_preds)  # shape (n_samples, n_clusters)
    feats = np.hstack([labels, dists])
    names = ["kmeans_label"] + [f"kmeans_dist_{i}" for i in range(n_clusters)]
    return feats, names

def compute_log_stats(oof_preds, eps=1e-12):
    """
    Mean and std of log(p+eps) per sample.
    """
    lp = np.log(oof_preds + eps)
    mean_ = lp.mean(axis=1, keepdims=True)
    std_  = lp.std(axis=1, keepdims=True)
    feats = np.hstack([mean_, std_])
    names = ["logp_mean", "logp_std"]
    return feats, names

def compute_pairwise_ratios(oof_preds):
    """
    Compute p_i / p_j for all i<j.
    """
    n, C = oof_preds.shape
    pairs = list(combinations(range(C), 2))
    ratios = np.stack([oof_preds[:, i] / (oof_preds[:, j] + 1e-12)
                       for i,j in pairs], axis=1)
    names = [f"ratio_{i}_{j}" for i,j in pairs]
    return ratios, names
def compute_second_order_diffs(oof_preds):
    """
    Compute second-order adjacent diffs:
      diff2_i = (p[i] - p[i+1]) - (p[i+1] - p[i+2])

    Returns
    -------
    feats : np.ndarray, shape (n_samples, C-2)
    names : list of str
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    first_diff = sorted_p[:, :-1] - sorted_p[:, 1:]
    # now adjacent diffs on first_diff
    C1 = first_diff.shape[1]
    second = first_diff[:, :-1] - first_diff[:, 1:]
    names = [f"diff2_{i}_{i+2}" for i in range(C1-1)]
    return second, names
def compute_autocorr_features(oof_preds, nlags=5):
    """
    Compute autocorrelation at lags 1..nlags for each sample.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, nlags)
    names : ['autocorr_1', ...]
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    n, C = sorted_p.shape
    feats = np.zeros((n, nlags))
    names = []
    for lag in range(1, nlags+1):
        names.append(f"autocorr_{lag}")
        # normalize each row
        for i,row in enumerate(sorted_p):
            row = row - row.mean()
            denom = np.dot(row, row)
            num = np.dot(row[:-lag], row[lag:])
            feats[i, lag-1] = num/denom if denom!=0 else 0.0
    return feats, names


def compute_segment_stats(oof_preds, n_segments=5):
    """
    Split sorted_preds into n_segments equal parts and compute mean/std per segment.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, 2*n_segments)
    names : ['seg0_mean','seg0_std',...]
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    n, C = sorted_p.shape
    seg_size = C//n_segments
    feats = []
    names = []
    for s in range(n_segments):
        start = s*seg_size
        end = (s+1)*seg_size if s<n_segments-1 else C
        seg = sorted_p[:, start:end]
        feats.append(seg.mean(axis=1, keepdims=True))
        feats.append(seg.std(axis=1, keepdims=True))
        names += [f"seg{s}_mean", f"seg{s}_std"]
    feats = np.hstack(feats)
    return feats, names

def compute_peak_stats(oof_preds, height=None, distance=1):
    """
    Find peaks in sorted_preds: return peak count, mean height, mean width.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, 3)
    names : ['peak_count','peak_mean_height','peak_mean_width']
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    n, C = sorted_p.shape
    pcs = np.zeros((n,3))
    for i,row in enumerate(sorted_p):
        peaks,_ = find_peaks(row, height=height, distance=distance)
        pcs[i,0] = len(peaks)
        pcs[i,1] = row[peaks].mean() if len(peaks)>0 else 0
        if len(peaks)>0:
            widths = peak_widths(row, peaks, rel_height=0.5)[0]
            pcs[i,2] = widths.mean()
    return pcs, ['peak_count','peak_mean_height','peak_mean_width']


def compute_ar_coeffs(oof_preds, lags=3):
    """
    Fit AR(lags) model to sorted_preds and return coefficients 1..lags.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, lags)
    names : ['arcoef_1',...]
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    n, C = sorted_p.shape
    feats = np.zeros((n, lags))
    for i,row in enumerate(sorted_p):
        try:
            model = AutoReg(row, lags=lags, old_names=False).fit()
            coeffs = model.params[1:]  # skip intercept
        except Exception:
            coeffs = np.zeros(lags)
        feats[i] = coeffs
    names = [f"arcoef_{j}" for j in range(1, lags+1)]
    return feats, names
# optionally, explicitly declare what’s exported on “from … import *”
# prediction_features.py
def compute_multi_stride_diffs(oof_preds):
    """
    Compute first-order diffs for all strides 1..C-1:
      diff_s_i = sorted_preds[:, i] - sorted_preds[:, i+s]

    Returns
    -------
    feats : np.ndarray, shape (n_samples, sum_{s=1..C-1}(C-s))
    names : list of str, e.g. ['diff_s1_0_1',...,'diff_s1_33_34',
                                'diff_s2_0_2',...,'diff_s2_32_34', …]
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    n, C = sorted_p.shape
    blocks = []
    names = []
    for s in range(1, C):
        # for each stride s, compute diffs between i and i+s
        diffs = np.stack([sorted_p[:, i] - sorted_p[:, i+s] for i in range(C-s)], axis=1)
        blocks.append(diffs)
        names += [f"diff_s{s}_{i}_{i+s}" for i in range(C-s)]
    feats = np.hstack(blocks)
    return feats, names

def compute_adj_diff_histogram(oof_preds, nbins=10):
    """
    For each sample, compute adjacent diffs of sorted preds (stride=1),
    then histogram those 34 diffs into `nbins` equal-width bins.

    Returns
    -------
    feats : np.ndarray, shape (n_samples, nbins)
      counts of adjacent diffs falling into each bin
    names : list of str, length nbins
      ['adj_diff_bin0', ..., 'adj_diff_bin{nbins-1}']
    """
    # compute stride-1 diffs
    sorted_p = -np.sort(-oof_preds, axis=1)
    diffs = sorted_p[:, :-1] - sorted_p[:, 1:]      # shape (n_samples, C-1)
    n, C1 = diffs.shape

    # global min/max across all samples & positions
    global_min = diffs.min()
    global_max = diffs.max()

    # build bin edges
    bin_edges = np.linspace(global_min, global_max, nbins+1)

    # histogram counts per sample
    counts = np.zeros((n, nbins), dtype=int)
    for b in range(nbins):
        lo, hi = bin_edges[b], bin_edges[b+1]
        mask = (diffs >= lo) & (diffs < hi)
        counts[:, b] = mask.sum(axis=1)

    names = [f"adj-his_diff_bin{b}" for b in range(nbins)]
    return counts, names


def compute_third_order_diffs(oof_preds):
    """
    Compute third-order diffs:
      d1_i  = p[i]   - p[i+1]
      d2_i  = d1_i   - d1_{i+1}
      d3_i  = d2_i   - d2_{i+1}
    Returns feats shape (n_samples, C-3) and names ['diff3_0_3',...]
    """
    # 先排序
    sorted_p = -np.sort(-oof_preds, axis=1)
    # 一阶差分
    d1 = sorted_p[:, :-1] - sorted_p[:, 1:]            # (n, C-1)
    # 二阶差分
    d2 = d1[:, :-1] - d1[:, 1:]                        # (n, C-2)
    # 三阶差分
    d3 = d2[:, :-1] - d2[:, 1:]                        # (n, C-3)
    C3 = d3.shape[1]
    names = [f"diff3_{i}_{i+3}" for i in range(C3)]
    return d3, names

def compute_log_adjacent_diffs(oof_preds, eps=1e-12):
    """
    Compute log differences between adjacent sorted preds:
      ld_i = log(p[i]+eps) - log(p[i+1]+eps)
    Returns shape (n_samples, C-1)
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    logp = np.log(sorted_p + eps)
    ld = logp[:, :-1] - logp[:, 1:]
    names = [f"logdiff_{i}_{i+1}" for i in range(ld.shape[1])]
    return ld, names

def compute_relative_diffs(oof_preds, eps=1e-12):
    """
    Compute relative change between adjacent sorted preds:
      rd_i = (p[i] - p[i+1]) / (p[i+1] + eps)
    Returns shape (n_samples, C-1)
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    num = sorted_p[:, :-1] - sorted_p[:, 1:]
    den = sorted_p[:, 1:] + eps
    rd = num / den
    names = [f"reldiff_{i}_{i+1}" for i in range(rd.shape[1])]
    return rd, names

def compute_diff_ratio_of_diffs(oof_preds, eps=1e-12):
    """
    Compute ratio of first-order diffs:
      dr_i = (d1_i) / (d1_{i+1} + eps)
    where d1 is adjacent diff.
    Returns shape (n_samples, C-2)
    """
    sorted_p = -np.sort(-oof_preds, axis=1)
    d1 = sorted_p[:, :-1] - sorted_p[:, 1:]
    num = d1[:, :-1]
    den = d1[:, 1:] + eps
    dr = num / den
    names = [f"dratio_{i}_{i+2}" for i in range(dr.shape[1])]
    return dr, names

__all__ = [
    "compute_entropy",
    "compute_adjacent_diffs",
    "compute_pairwise_diff",
    "compute_dispersion",
    "compute_lastn_stats_multi",
    "compute_topn_stats_multi",
    "compute_median_mad",
    "compute_skew_kurt",
    "compute_percentile_iqr",
    "compute_renyi_entropy",
    "compute_kl_uniform",
    "compute_js_uniform",
    "compute_hellinger_uniform",
    "compute_mass_topk",
    "compute_tail_mass",
    "compute_cdf_slope",
    "compute_pca_components",
    "compute_kmeans_features",
    "compute_log_stats",
    "compute_pairwise_ratios",
    "compute_adj_diff_histogram",
    "compute_multi_stride_diffs",
    "compute_peak_stats",
    "compute_segment_stats",
    "compute_ar_coeffs",
    "compute_autocorr_features",
    "compute_second_order_diffs",
    "compute_third_order_diffs",
    "compute_log_adjacent_diffs",
    "compute_relative_diffs",
    "compute_diff_ratio_of_diffs"
]
