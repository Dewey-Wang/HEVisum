import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# ✅ 安全轉換：避免 ToTensor() 對 Tensor 失效
class SafeTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img  # 已是 Tensor，不再轉換
        return self.transform(img)

class importDataset(Dataset):
    def __init__(self, center_tile, subtiles, neighbor_tiles, label,
                 meta=None,
                 node_feats=None, adj_lists=None, edge_feats=None,
                 transform=None):

        self.center_tile = center_tile
        self.subtiles = subtiles
        self.neighbor_tiles = neighbor_tiles
        self.label = label
        self.meta = meta if meta is not None else [None] * len(center_tile)

        # 👇 加入 graph features
        self.node_feats = node_feats
        self.adj_lists = adj_lists
        self.edge_feats = edge_feats

        self.transform = transform or SafeTransform()

    def __len__(self):
        return len(self.center_tile)

    def __getitem__(self, idx):
        # 🖼️ 圖像部分
        center_tile = self.transform(self.center_tile[idx])
        subtiles = torch.stack([self.transform(tile) for tile in self.subtiles[idx]])
        neighbor_tiles = torch.stack([self.transform(tile) for tile in self.neighbor_tiles[idx]])
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        # 🗺️ meta (coords)
        meta = self.meta[idx]
        if isinstance(meta, np.ndarray):
            meta = torch.tensor(meta, dtype=torch.float32)
        elif isinstance(meta, list):
            meta = torch.tensor(meta, dtype=torch.float32)
        elif isinstance(meta, torch.Tensor):
            meta = meta.to(torch.float32)

        # ➕ Graph features
        max_neighbors = 7  # or whatever k you use
        adj_list = self.adj_lists[idx] if self.adj_lists is not None else []

        # 轉成固定長度的 Tensor
        adj_array = np.zeros((max_neighbors, 2), dtype=np.float32)
        for i, (j, w) in enumerate(adj_list[:max_neighbors]):
            adj_array[i] = [j, w]

        # 🛠️ edge_feat 轉型處理
        edge_feat_i = self.edge_feats[idx]
        if isinstance(edge_feat_i, np.ndarray) and edge_feat_i.dtype == object:
            edge_feat_i = np.array(edge_feat_i.tolist(), dtype=np.float32)
        elif isinstance(edge_feat_i, list):
            edge_feat_i = np.array(edge_feat_i, dtype=np.float32)

        graph_feats = {
            'node_feat': torch.tensor(self.node_feats[idx], dtype=torch.float32) if self.node_feats is not None else None,
            'adj_list': torch.tensor(adj_array, dtype=torch.float32),
            'edge_feat': torch.tensor(edge_feat_i, dtype=torch.float32) if edge_feat_i is not None else None,
        }


        return {
            'center_tile': center_tile,
            'subtiles': subtiles,
            'neighbor_tiles': neighbor_tiles,
            'label': label,
            'meta': meta,
            **graph_feats  # ➕ 合併進 batch dictionary
        }