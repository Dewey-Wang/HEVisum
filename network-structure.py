import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DeepSpotModel(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=35, feature_dim=128):
        super().__init__()

        # 共享的 CNN encoder（輸出 feature vector）
        self.encoder = self.build_encoder(backbone, feature_dim)

        # 預測頭
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def build_encoder(self, backbone, feature_dim):
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=None)
            layers = list(resnet.children())[:-2]  # 拿到 conv5_x 前
            cnn = nn.Sequential(*layers)
            gap = nn.AdaptiveAvgPool2d(1)
            return nn.Sequential(cnn, gap, nn.Flatten(), nn.Linear(512, feature_dim))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def encode_tile(self, x):
        return self.encoder(x)  # 輸出 (B, feature_dim)

    def forward(self, tile, subtiles, neighbor_tiles):
        # tile: [B, 3, H, W]
        # subtiles: [B, 9, 3, H', W']
        # neighbor_tiles: [B, N, 3, H, W] (N = 8)

        B = tile.shape[0]

        # 1. Spot tile → CNN
        f_center = self.encode_tile(tile)  # (B, D)

        # 2. Sub-tiles
        subtiles = subtiles.view(B * 9, *subtiles.shape[2:])      # (B*9, 3, H', W')
        f_subtiles = self.encode_tile(subtiles).view(B, 9, -1)     # (B, 9, D)
        f_sub = torch.mean(f_subtiles, dim=1)                     # (B, D)

        # 3. Neighbor tiles
        neighbor_tiles = neighbor_tiles.view(B * 8, *neighbor_tiles.shape[2:])
        f_neighbors = self.encode_tile(neighbor_tiles).view(B, 8, -1)
        f_neighbor = torch.max(f_neighbors, dim=1).values         # (B, D)

        # Concatenate all
        f = torch.cat([f_center, f_sub, f_neighbor], dim=1)       # (B, D*3)
        out = self.fc(f)                                          # (B, 35)

        return out
