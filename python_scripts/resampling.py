import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List
from python_scripts.aug import AugmentFn
import torch
import math
def oversample_and_augment_grouped_data_by_topk(
    grouped_data: dict,
    rank_cols: List[str],
    K: int = 1,
    N_max: int = 100000,
    fill_to_n_max: bool = False,
    image_keys: List[str] = ['tile','subtiles'],
    slide_key: str = 'slide_idx',
    label_key: str = 'label',
    random_state: int = 42
) -> dict:
    """
    对 grouped_data 做 Top-K 组合的 capped over-/under-sampling，
    并对多采样出来的样本立刻调用 AugmentFn 再做一次增强，
    返回新的 grouped_data dict，直接可传给 importDataset。
    """

    # 1) 把所有 label 转成 numpy array 再堆成 (N, C) lab_mat
    lab_list = []
    for lab in grouped_data[label_key]:
        if isinstance(lab, torch.Tensor):
            lab_list.append(lab.detach().cpu().numpy())
        else:
            lab_list.append(np.array(lab))
    lab_mat = np.stack(lab_list, axis=0)  # shape (N, C)

    N, C = lab_mat.shape
    # 2) 计算每行 Top-K 的列索引
    #    argsort 升序，取后 K，再翻序成降序
    sorted_idx = np.argsort(lab_mat, axis=1)
    topk_idx   = sorted_idx[:, ::-1][:, :K]  # shape (N, K)

    # 3) 用 rank_cols 映射成 top1..topK 名称
    topk_names = np.array(rank_cols)[topk_idx]  # shape (N, K), dtype=str

    # 4) 构造一个 DataFrame 用于分组
    df = pd.DataFrame({
        slide_key: grouped_data[slide_key],
        **{f"top{i+1}": topk_names[:, i] for i in range(K)}
    })
    label_cols = [f"top{i+1}" for i in range(K)]
    # 5) 统计每组大小 & 目标 size
    grp_sizes = df.groupby(label_cols).size().sort_values(ascending=False)   # 从大到小；如果想从小到大，设为 True

    n_groups  = len(grp_sizes)
    max_size = int(grp_sizes.max())
        # 理想情况下每组目标大小：
    base_target = N_max // n_groups
        
        # 如果 base_target > max_size，需要根据 fill_to_n_max 决定
    if base_target <= max_size:
        target = base_target
    else:
        if fill_to_n_max:
            target = math.ceil(N_max / n_groups)
        else:
            target = max_size

    print(f"每组目标大小：{target}，最大组大小：{max_size}，总样本数：{N}，分组数：{n_groups}")
    # 6) over/under 分组采样，记录 keep 和 dup
    rng = np.random.RandomState(random_state)
    keep_idx = []
    dup_idx  = []
    for combo, cnt in grp_sizes.items():
        # 强制把 combo 变成 tuple，K>1 时本来就是 tuple，K=1 时变成 (scalar,)
        if not isinstance(combo, tuple):
            combo = (combo,)

        # 下面的逻辑就不用改
        mask = np.ones(N, dtype=bool)
        for col, val in zip(label_cols, combo):
            mask &= (df[col].values == val)
        idxs = np.nonzero(mask)[0]

        if cnt < target:
            keep_idx.append(idxs)
            extra = rng.choice(idxs, target - cnt, replace=True)
            dup_idx.append(extra)
        else:
            chosen = rng.choice(idxs, target, replace=False)
            keep_idx.append(chosen)


    keep_idx = np.concatenate(keep_idx)
    dup_idx  = np.concatenate(dup_idx) if dup_idx else np.array([], dtype=int)

    # 7) 重建 grouped_data：先保留 keep，再对 dup 做 AugmentFn
    augmenter = AugmentFn(repeats=1)
    new_data  = {k: [] for k in grouped_data}

    # 7a) 保留
    for i in keep_idx:
        for k, lst in grouped_data.items():
            new_data[k].append(lst[int(i)])

    print(f"Starting to augment {len(dup_idx)} duplicated samples...")
    # 7b) 对 dup_idx 增强
    for i in dup_idx:
        sample = {k: deepcopy(grouped_data[k][int(i)]) for k in grouped_data}
        aug    = augmenter(sample, base_idx=int(i), aug_idx=0)
        for k in grouped_data:
            # 只有 image_keys 做替换，其他 key 重用原样
            if k in image_keys:
                new_data[k].append(aug[k])
            else:
                new_data[k].append(grouped_data[k][int(i)])

    return new_data


def check_source_idx_consistency(train_aug: dict, random_state: int = None):
    """
    检查 train_aug 中 'source_idx' 的重复情况，并验证重复样本的 'label' 和 'tile' 是否一致。
    
    参数：
        train_aug (dict): 包含至少 'source_idx', 'label', 'tile' 键的字典。
        random_state (int, optional): 随机种子，用于可复现地选择重复项。默认不设随机种子。
    
    返回：
        dict: 包含以下字段：
            - n_unique (int): 不同 source_idx 的数量
            - dup_vals (np.ndarray): 所有出现超过一次的 source_idx 值
            - selected (int or None): 随机选中的重复 source_idx（如果没有重复则为 None）
            - count_selected (int or None): 选中值出现的次数
            - positions (np.ndarray or None): 该值在 train_aug 中的位置索引
            - all_labels_same (bool or None): 这些样本的 label 是否全部一致
            - all_tiles_same (bool or None): 这些样本的 tile 是否全部一致
    """
    # 设置随机种子
    if random_state is not None:
        random.seed(random_state)
    
    # 转为 numpy 数组
    source_idx_arr = np.array(train_aug['source_idx'])
    
    # 统计 unique 和 counts
    unique_vals, counts = np.unique(source_idx_arr, return_counts=True)
    print("一共有多少不同的 source_idx：", len(unique_vals))
    dup_vals = unique_vals[counts > 1]
    print("出现重复的 source_idx 有：", dup_vals)
    
    # 如果没有重复
    if len(dup_vals) == 0:
        print("没有重复的 source_idx，结束检查。")
        return {
            'n_unique': len(unique_vals),
            'dup_vals': dup_vals,
            'selected': None,
            'count_selected': None,
            'positions': None,
            'all_labels_same': None,
            'all_tiles_same': None
        }
    
    # 随机选一个重复值
    sel = random.choice(dup_vals.tolist())
    count_sel = int(counts[unique_vals == sel][0])
    print(f"随机抽到的重复 source_idx：{sel}，出现次数：{count_sel}")
    
    # 找到所有对应位置
    positions = np.where(source_idx_arr == sel)[0]
    print("在 train_aug 中的位置索引：", positions)
    
    # 取出对应 label 和 tile
    labels = [train_aug['label'][i] for i in positions]
    tiles = [train_aug['tile'][i] for i in positions]
    
    # 转为 numpy 比较
    labels_np = [(lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.array(lab)) for lab in labels]
    all_labels_same = all(np.array_equal(labels_np[0], lab) for lab in labels_np)
    print("这些样本的 label 是否完全一致？", all_labels_same)
    
    tiles_np = [(t.detach().cpu().numpy() if hasattr(t, "detach") else np.array(t)) for t in tiles]
    all_tiles_same = all(np.array_equal(tiles_np[0], t) for t in tiles_np)
    print("这些样本的 tile 是否完全一致？", all_tiles_same)
    
    return {
        'n_unique': len(unique_vals),
        'dup_vals': dup_vals,
        'selected': sel,
        'count_selected': count_sel,
        'positions': positions,
        'all_labels_same': all_labels_same,
        'all_tiles_same': all_tiles_same
    }