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

# ✅ Dataset 定義
class importDataset(Dataset):
    def __init__(self, center_tile, subtiles, neighbor_tiles, label, meta=None, transform=None):
        self.center_tile = center_tile
        self.subtiles = subtiles
        self.neighbor_tiles = neighbor_tiles
        self.label = label
        self.meta = meta if meta is not None else [None] * len(center_tile)

        self.transform = transform or SafeTransform()

    def __len__(self):
        return len(self.center_tile)

    def __getitem__(self, idx):
        # ✅ 安全轉換中心 tile
        center_tile = self.transform(self.center_tile[idx])  # shape: (3, H, W)

        # ✅ Subtiles: List of (H, W, 3)
        subtiles = torch.stack([self.transform(tile) for tile in self.subtiles[idx]])  # (9, 3, h, w)

        # ✅ Neighbor tiles: List of (H, W, 3)
        neighbor_tiles = torch.stack([self.transform(tile) for tile in self.neighbor_tiles[idx]])  # (8, 3, H, W)

        # ✅ Label
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        # ✅ Normalized coordinates (x, y)
        meta = self.meta[idx]  # already normalized
        if isinstance(meta, np.ndarray):
            meta = tuple(meta.tolist())  # convert to tuple
        elif isinstance(meta, list):
            meta = tuple(meta)
        elif isinstance(meta, torch.Tensor):
            meta = tuple(meta.cpu().numpy().tolist())

        return {
            'center_tile': center_tile,
            'subtiles': subtiles,
            'neighbor_tiles': neighbor_tiles,
            'label': label,
            'meta': meta
        }
